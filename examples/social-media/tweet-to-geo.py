import sys
import json

fname = sys.argv[1]

with open(fname, 'r') as f:
    tweets = [json.loads(line.strip()) for line in f]

geojson = dict(type='FeatureCollection', features=[])
for tweet in tweets:
    geo = tweet.get('geo')
    if geo:
        geo['coordinates'].reverse()
        feature = dict(type='Feature', geometry=geo, properties=tweet)
        geojson['features'].append(feature)

with open(sys.argv[2], 'w') as w:
    w.write(json.dumps(geojson))
