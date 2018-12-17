**eo-learn makes extraction of valuable information from satellite imagery easy.**

The availability of open Earth observation (EO) data through the Copernicus and Landsat programs represents an
unprecedented resource for many EO applications, ranging from ocean and land use and land cover monitoring,
disaster control, emergency services and humanitarian relief. Given the large amount of high spatial resolution
data at high revisit frequency, techniques able to automatically extract complex patterns in such _spatio-temporal_
data are needed.

**eo-learn** is a collection of open source Python packages that have been developed to seamlessly access and process
_spatio-temporal_ image sequences acquired by any satellite fleet in a timely and automatic manner. **eo-learn** is
easy to use, it's design modular, and encourages collaboration -- sharing and reusing of specific tasks in a typical
EO-value-extraction workflows, such as cloud masking, image co-registration, feature extraction, classification, etc. Everyone is free
to use any of the available tasks and is encouraged to improve the, develop new ones and share them with the rest of the community.

**eo-learn** makes extraction of valuable information from satellite imagery as easy as defining a sequence of operations to be performed on satellite imagery. Image below illustrates a processing chain that maps water in satellite imagery by thresholding the Normalised Difference Water Index in user specified region of interest.

![eo-learn-workflow0illustration](figures/eo-learn-illustration.png)

**eo-learn** _library acts as a bridge between Earth observation/Remote sensing field and Python ecosystem for data science and machine learning._ The library is written in Python and uses NumPy arrays to store and handle remote sensing data. Its aim is to make entry easier for non-experts to the field of remote sensing on one hand and bring the state-of-the-art tools for computer vision, machine learning, and deep learning existing in Python ecosystem to remote sensing experts.

## Package Overview

**eo-learn** is divided into several subpackages according to different functionalities and external package dependencies. Therefore it is not necessary for user to install entire package but only the parts that he needs.

At the moment there are the following subpackages:

- **eo-learn-core** - The main subpackage which implements basic building blocks (EOPatch, EOTask and EOWorkflow) and commonly used functionalities.
- **eo-learn-coregistration** - The subpackage that deals with image co-registraion.
- **eo-learn-features** - A collection of utilities for extracting data properties and feature manipulation.
- **eo-learn-geometry** - Geometry subpackage used for geometric transformation and conversion between vector and raster data.
- **eo-learn-io** - Input/output subpackage that deals with obtaining data from Sentinel Hub services or saving and loading data locally.
- **eo-learn-mask** - The subpackage used for masking of data and calculation of cloud masks.
- **eo-learn-ml-tools** - Various tools that can be used before or after the machine learning process.

## Building blocks

The design of the **eo-learn** library follows the dataflow programing paradigm and consists of three building blocks:
1. **EOPatch**
    * common data-object for _spatio-temporal_ EO and non-EO data, and their derivatives
2. **EOTask**
    * a single, well-defined action being performed on existing **EOPatch**(es)
    * each **EOTask** takes an **EOPatch** as an input and returns a modified **EOPatch**
3. **EOWorkflow**
    * a collection of **EOTask**s that together represent an _EO-value-adding-processing_ chain
4. **EOExecutor**
    * Handles execution of a **EOWorkflow** multiple times at once with different input data.

### EOPatch

**EOPatch** contains multi-temporal remotely sensed data of a single patch (area) of Earth's surface typically
defined by a bounding box in specific coordinate reference system. The **EOPatch** object can also be used as a placeholder
for all quantities, either derived from the satellite imagery or from some other external source, such as for example
biophysical indices, ground truth reference data, weather data, etc. **EOPatch** is completely sensor-agnostic, meaning that imagery from different sensors (satellites) or sensor types (optical, synthetic-aperture radar, ...) can be added to an **EOPatch**.

### EOTask

**EOTask**s are in a sense the heart of the eo-learn library. They define in what way the available satellite imagery can be manipulated in order to extract valuable information out of it. Typical user will most often just be interested in what kind of tasks are already implemented and ready to be used or how to implement an **EOTask**, if it doesn't exist yet.

### EOWorkflow

**EOWorkflow** represents the entire EO processing chain or EO-value-extraction pipeline by chaining or connecting sequence of **EOTask**s. The **EOWorkflow** takes care that the **EOTask**s are executed in the correct order and with correct parameters. **EOWorkflow** is executed on a single **EOPatch** at a time, but the same **EOWorkflow** can be executed on multiple **EOTask**s.

Under the hood the **EOWorkflow** builds a directed acyclic graph (in case user tries to build a cyclic graph the **EOWorkflow** will complain). There's no limitation on the number of nodes (**EOTask**s with inputs) or the graph's topology. The **EOWorkflow** first names the input tasks that persist over executions, determines the ordering of the tasks, executes the task in that order, and finally returns the results of tasks with no outgoing edge.

### EOExecutor

**EOExecutor** wraps the entire pipeline process together by executing a **EOWorkflow** as many times as required, each time with different user-defined input parameters. The executions can be parallel or consecutive. During each execution process **EOExecutor** monitors the progress, saves log files and catches any possible pipeline breaking errors. At the end it can produce a report with all information about executions. It provides an easy way to track when something was processed and how to repeat the process with exactly the same parameters.
