import lightgbm


PRECISION_SCORES = 4
PRECISION_THRESHOLD = None
MAX_DN = 10000
EVALSCRIPT_VERSION = 3
DIGITAL_NUMBER = False


def parse_subtree(node, brackets=True):
    if 'leaf_index' in node:
        score = float(node["leaf_value"])
        if PRECISION_SCORES is not None:
            score = round(score, PRECISION_SCORES)
        return f'{score}'
    
    band = BANDS[int(node["split_feature"])]
    
    threshold = float(node["threshold"])
    if PRECISION_THRESHOLD is not None:
        threshold = round(threshold, PRECISION_THRESHOLD)
    if DIGITAL_NUMBER:
        threshold = int(MAX_DN * threshold)

    condition = f'{band}{node["decision_type"]}{threshold}'
    
    left = parse_subtree(node['left_child'])
    right = parse_subtree(node['right_child'])
    
    result = f'({condition})?{left}:{right}'
    if brackets:
        return f'({result})'
    return result

def parse_one_tree(root, index):
    return \
f"""function pt{index}({BANDS_STR}) {{ 
return {parse_subtree(root, brackets=False)};
}}
"""

def parse_trees(trees):
    
    sample_str = ','.join(f'sample.{band}' for band in BANDS)
    
    tree_functions = '\n'.join([parse_one_tree(tree['tree_structure'], idx)
                                  for idx, tree in enumerate(trees)])
    function_sum = '+'.join([f'pt{i}({BANDS_STR})' for i in range(len(trees))])
    
    bands_array = ','.join(f'"{band}"' for band in BANDS)
    
    input_units = 'DN' if DIGITAL_NUMBER else 'reflectance'
    
    if EVALSCRIPT_VERSION < 3:
        return \
f"""
{tree_functions}
function predict({BANDS_STR}) {{ 
    return [{function_sum}];
}}
return [predict({BANDS_STR})];
"""
    return f"""
//VERSION={EVALSCRIPT_VERSION}
function setup() {{
    return {{
        input: [{{
            bands: [{bands_array}],
            units: "{input_units}"
        }}],
        output: {{
            id:"default",
            bands: 1,
            sampleType: "FLOAT32"
        }}
    }}
}}
function evaluatePixel(sample) {{
    return [predict({sample_str})]
}}
{tree_functions}
function predict({BANDS_STR}) {{ 
    return [1/(1+Math.exp(-1*({function_sum})))];
}}
"""

def parse_model(model_filename, output_filename):
    model = lightgbm.Booster(model_file=model_filename)
    
    model_json = model.dump_model()
    
    model_javascript = parse_trees(model_json['tree_info'])
    
    with open(output_filename, 'w') as f:
        f.write(model_javascript)
        
    return model_javascript
