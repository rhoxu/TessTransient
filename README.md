# TessTransient: A transient event detection pipeline for TESS.

This pipeline utilises the MAST archive and TESSreduce (see https://github.com/CheerfulUser/TESSreduce) to search for poorly localised transient events in TESS data. 

Input an estimation for an event's right ascension and declination, and an associated one dimensional position error, as well as the time of the event. 

```
event = <span style='color: red;'>TessTransient</span>(ra=277.48, dec=61.78, eventtime=58951.381019, error=2, path='../../SampleTessTransient/', eventname='GRB200412B')
```
