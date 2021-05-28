import matplotlib.pyplot as plt
import datetime as dt
import numpy as np

from sentinelhub import (SHConfig, CRS, BBox, DataCollection, SentinelHubRequest,
                         MimeType, bbox_to_dimensions)

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


def visualize(patch, factor=3.5):
    fig, ax = plt.subplots(ncols=3, figsize=(22,7))
    ax[0].imshow(factor*patch.data['BANDS-S2-L1C'][0][...,[3,2,1]].squeeze())
    ax[0].set_title(f'True color, {patch.timestamp[0]}')

    ax[1].imshow(patch.data_timeless['DEM'].squeeze())
    ax[1].set_title('DEM')

    ax[2].imshow(patch.mask_timeless['water_label'].squeeze(), vmin=0, vmax=1)
    ax[2].set_title('water mask')
    

def predict_on_sh(model_script, bbox, size, timestamp, config):
    request = SentinelHubRequest(
        evalscript=model_script,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL2_L1C,
                time_interval=(timestamp-dt.timedelta(minutes=5),timestamp+dt.timedelta(minutes=5)),
                maxcc=1,
            )
        ],
        responses=[SentinelHubRequest.output_response('default',MimeType.TIFF)],
        bbox=bbox,
        size=size,
        config=config
    )
    return request.get_data()[0]


def get_predictions(patch, model, config):
    model_script = parse_model(model, None)
    sh_prediction = predict_on_sh(model_script, patch.bbox, (64, 64), patch.timestamp[0], config)
    
    features = patch.data['FEATURES'][0]
    f_s = features.shape
    model_prediction = model.predict_proba(features.reshape(f_s[0]*f_s[1],f_s[2]))[...,-1].reshape(f_s[0],f_s[1])

    return features[..., [2, 1, 000]], sh_prediction, model_prediction


def plot_comparison(rgb_array, sh_prediction, model_prediction, threshold=0.5):
    fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(24, 12), sharex=True, sharey=True)
    
    for axx in ax.flatten():
        axx.set_xticks([])
        axx.set_yticks([])

    ax[0][0].imshow(3.5*rgb_array)
    ax[0][0].set_title('RGB image')

    ax[0][1].imshow(sh_prediction, vmin=0, vmax=1)
    ax[0][1].set_title('[a] water prediction probabilities with evalscript on SH', fontsize=14)
    
    ax[0][2].imshow(model_prediction, vmin=0, vmax=1)
    ax[0][2].set_title('[b] water prediction probabilities with model, locally', fontsize=14)
    
    ax[0][3].imshow(sh_prediction - model_prediction, vmin=-0.2, vmax=0.2, cmap="RdBu")
    ax[0][3].set_title('differences between [a] and [b]', fontsize=14)
    
    sh_thr = np.where(sh_prediction>threshold, 1, 0)
    ax[1][0].imshow(3.5*rgb_array)
    ax[1][0].set_title('RGB image')

    ax[1][1].imshow(sh_thr, vmin=0, vmax=1)
    ax[1][1].set_title('[c] water prediction with evalscript on SH', fontsize=14)
    
    model_thr = np.where(model_prediction>threshold, 1, 0)
    ax[1][2].imshow(model_thr, vmin=0, vmax=1)
    ax[1][2].set_title('[d] water prediction with model, locally', fontsize=14)
    
    ax[1][3].imshow(sh_thr - model_thr, vmin=-1, vmax=1, cmap="RdBu")
    ax[1][3].set_title('differences between [d] and [d]', fontsize=14)
    
    plt.tight_layout(pad=0.05)
