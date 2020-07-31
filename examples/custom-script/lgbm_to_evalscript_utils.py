PRECISION_SCORES = 4
PRECISION_THRESHOLD = None

BANDS = ['B02', 'B03', 'B04', 'B08', 'B11']
BANDS_STR = ','.join(BANDS)
MODEL_INPUTS = ['B02', 'B03', 'B04', 'NDWI', 'NDMI']
MODEL_INPUTS_STR = ', '.join(MODEL_INPUTS)

def parse_subtree(node, brackets=True):
    if 'leaf_index' in node:
        score = float(node["leaf_value"])
        if PRECISION_SCORES is not None:
            score = round(score, PRECISION_SCORES)
        return f'{score}'
    
    feature = MODEL_INPUTS[int(node["split_feature"])]
    
    threshold = float(node["threshold"])
    if PRECISION_THRESHOLD is not None:
        threshold = round(threshold, PRECISION_THRESHOLD)
    
    condition = f'{feature}{node["decision_type"]}{threshold}'
    
    left = parse_subtree(node['left_child'])
    right = parse_subtree(node['right_child'])
    
    result = f'({condition})?{left}:{right}'
    if brackets:
        return f'({result})'
    return result

def parse_one_tree(root, index):
    return \
f"""
function pt{index}({MODEL_INPUTS_STR}) {{ 
   return {parse_subtree(root, brackets=False)};
}}
"""

def parse_trees(trees):
    
    tree_functions = '\n'.join([parse_one_tree(tree['tree_structure'], idx)
                                  for idx, tree in enumerate(trees)])
    function_sum = '+'.join([f'pt{i}({MODEL_INPUTS_STR})' for i in range(len(trees))])
    
    return f"""
//VERSION=3

function setup() {{
    return {{
        input: [{{
            bands: [{','.join(f'"{band}"' for band in BANDS)}],
            units: "reflectance"
        }}],
        output: {{
            id:"default",
            bands: 1,
            sampleType: "FLOAT32"
        }}
    }}
}}

function evaluatePixel(sample) {{
    let NDWI = index(sample.B03, sample.B08);
    let NDMI = index(sample.B08, sample.B11);
    
    return [predict(sample.B02, sample.B03, sample.B04, NDWI, NDMI)]
}}

{tree_functions}

function predict({MODEL_INPUTS_STR}) {{ 
    return [1/(1+Math.exp(-1*({function_sum})))];
}}
"""

def parse_model(model, js_output_filename=None):
    model_json = model.booster_.dump_model()
    
    model_javascript = parse_trees(model_json['tree_info'])
    
    if js_output_filename:
        with open(js_output_filename, 'w') as f:
            f.write(model_javascript)
        
    return model_javascript


