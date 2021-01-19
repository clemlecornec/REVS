# REVS: Road transport Emissions and Variance Simulator

## Prerequisites

```
* Python 3.6.3
* Numpy 1.19.4
* Pandas 1.1.4
* Scipy 1.4.1
* chainConsumer 0.31.0
* matplotlib 3.2.1
```

## Getting Started

To run this project, install it locally. We suggest that you use the same folders structure than us, namely:
```
* [CoefficientsFinal](./CoefficientsFinal/)
* [CoefficientsTrained](./CoefficientsTrained/)
* [DataFinal](./DataFinal)
 * [ExtractedData](./DataFinal/ExtractedData)
   *[DE5](./DataFinal/ExtractedData/DE5) 
   *[DE6preRDE](./DataFinal/ExtractedData/DE6preRDE) 
   * etc,
 * [RawData](./DataFinal/RawData)
  * [DE5](./DataFinal/RawData/DE5)
   *[Audi_A3_2201928463] (./DataFinal/RawData/DE5/Audi_A3_2201928463)
    * [Audi_A3_2201928463.csv]
    * [specification.txt]
   *[Nissan_Qashqai_1238374] (./DataFinal/RawData/DE5/Nissan_Qashqai_1238374)
    * [Nissan_Qashqai_1238374.csv]
    * [specification.txt]
   * etc,
  * etc,
* REVS.py
* utils.py
* Vehicles.py
* testREVS.py
* testREVS.py
```

Please note that if you just wish to use the pre-trained coefficients and not train your own, these are available in the CoefficientsTrained folder.

## Structure of the data

Please note that the current code assumes the original datasets to contain the following columns: speed, altitude, Nox. If the altitude data is not available, we suggest making the assumption that it was equal to zero during the test. It also assumes that the specification.txt file available for all the vehicles to contain the following rows: ID, Manufacturer, Model, Fuel, Euro standard, Height, Width, Weight.

## Test the code

The freely available data from the UK Department for Transport can be used to test the code [DfTData](https://www.gov.uk/government/publications/vehicle-emissions-testing-programme-conclusions). Please note that this code available on this repository used the Euro 6 Vehicle Data as a training and testing files to ensure that the code was running properly. Name of the columns in the original datafile needs to be adapted, and the specification.text files to be created for each individual vehicle.

## Contributing

Please read CONTRIBUTING.md for details on our code of conduct, and the process for submitting pull requests to us.

## Authors

* **Clémence Le Cornec**

## Citation

Please cite as:
```bibtex
@article{LeCornec2021,
title = {REVS: Road transport Emissions and Variance Simulator, [in prep]},
author = {Clémence M. A. Le Cornec and Maarten van Reeuwijk and Nick Molden and Marc E. J. Stettler},
year = {2021}
```

## License

This project is licensed under the GNU GPLv3 - see the [GNUGPLv3](https://www.gnu.org/licenses/gpl-3.0.html) for details
