## Social Media Data

- Example Paris 2015 data from https://zenodo.org/record/819905
- Hydrated with code based on https://datorium.gesis.org/xmlui/handle/10.7802/1504

Get a full tweet object from an id (requires Twitter API Access)
```sh
$ API_KEY=[YOUR_KEY] ACCESS_TOKEN=[YOUR_TOKEN] python hydrate.py paris_geo_tweet_id.csv raw.ndjson
```

Convert to geojson
```sh
$ python tweet-to-geo.py raw.ndjson twitter_data.geojson
```
