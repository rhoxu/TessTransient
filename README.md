# TessTransient: A transient event detection pipeline for TESS.

This pipeline utilises the MAST archive and TESSreduce (see https://github.com/CheerfulUser/TESSreduce) to search for poorly localised transient events in TESS data. 

To get started, download this repository and navigate into it via a command terminal. Make sure you are in the home folder containing setup.py, and then run `pip install .` to install the package. Have a look at the example notebook `example.ipynb` - input an estimation for an event's right ascension and declination, and an associated one dimensional position error, as well as the time of the event. There are many functions and additional parameters to take into consideration; hopefully the docs in the `TessTransient` folder are commented well enough. 

If you have any questions whatsoever, flick me (Hugh) an email at hugh11rox@gmail.com!

To come: non detection limiting magnitudes etc..

