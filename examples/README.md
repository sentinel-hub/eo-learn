# eo-learn examples

This folder contains example Earth observation workflows that extract valuable information from satellite imagery.  

##### Requirements

In order to run the example you'll need a Sentinel Hub account. You can get a trial version [here](https://www.sentinel-hub.com).

Once you have the account set up, login to [Sentinel Hub Configurator](https://apps.sentinel-hub.com/configurator/). By default you will already have the default confoguration with an **instance ID** (alpha-numeric code of length 36). For these examples it is recommended that you create a new configuration (`"Add new configuration"`) and set the configuration to be based on **Python scripts template**. Such configuration will already contain all layers used in these examples. Otherwise you will have to define the layers for your  configuration yourself.

After you have decided which configuration to use, you have two options You can either put configuration's **instance ID** into `sentinelhub` package's configuration file following the [configuration instructions](http://sentinelhub-py.readthedocs.io/en/latest/configure.html) or you can write it down in the example notebooks.


