Streamflow Prediction in Tapi Basin
===
This project is a re-implementation of a scientific study and aims to reproduce the methodology and results of the following research:

<blockquote>
  <a href="https://doi.org/10.1007/s00500-021-05585-9"><strong>
    "Data-driven modelling framework for streamflow prediction in a physio-climatically heterogeneous river basin"</strong>
  </a>
  <div align="right">– Sharma et al., 2021</div>
</blockquote>


The study evaluates the performance of Model Trees (MT) for one-day-ahead streamflow forecasting at multiple stream gauging stations of a river exhibiting both intermittent and perennial flow characteristics within a physio-climatically heterogeneous basin.

Notebook Preview
---
A static HTML version of the notebook with outputs is available for reference:  

* [<strong>Streamflow Prediction</strong>](https://github.com/SrijanSiddharth/Streamflow-Tapi-Basin/blob/main/notebooks/streamflow_full.html)

Data
---
This project uses daily rainfall and streamflow data for two stream gauging stations (out of three studied in the original paper) within the Tapi basin. The original datasets were collected by official government agencies. Due to sharing restrictions, the data cannot be distributed publicly. Additionally, other important weather variables like temperature, humidity, wind speed, etc. were not available at the required timescale and therefore not included in this study.

Usage
---
#### 1. Install Python dependencies:
```bash
pip install -r requirements.txt
```
#### 2. Install Java
*  OpenJDK 11 (or later)

This project was tested with [JDK 17.0.12](https://www.oracle.com/java/technologies/javase/jdk17-archive-downloads.html)