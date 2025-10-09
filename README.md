<p align="center">
<img src=https://github.com/dav0dea/goofi-pipe/assets/36135990/60fb2ba9-4124-4ca4-96e2-ae450d55596d width="150">
</p>

<h1 align="center">goofi-pipe</h1>
<h3 align="center">Generative Organic Oscillation Feedback Isomorphism Pipeline</h3>

<p align="center">
  <a href="https://deepwiki.com/dav0dea/goofi-pipe"><img src="https://deepwiki.com/badge.svg" alt="Ask DeepWiki"></a>
  <a href="https://github.com/dav0dea/goofi-pipe/actions/workflows/pytest.yml"><img alt="GitHub Actions Workflow Status" src="https://img.shields.io/github/actions/workflow/status/dav0dea/goofi-pipe/pytest.yml"></a>
  <a href="https://pypi.org/project/goofi/"><img alt="PyPI - Version" src="https://img.shields.io/pypi/v/goofi"></a>
  <a href="https://github.com/dav0dea/goofi-pipe/blob/main/LICENSE"><img alt="GitHub License" src="https://img.shields.io/github/license/dav0dea/goofi-pipe"></a>
  <a href="https://pypi.org/project/goofi/"><img alt="PyPI - Python Version" src="https://img.shields.io/pypi/pyversions/goofi"></a>
</p>

<h3 align="center"><a href="https://notebooklm.google.com/notebook/568087b5-8687-4175-9429-eefba15440d9?artifactId=199781e4-6a1d-4e9f-a4a7-6ae71edf02a8">AI Podcast about goofi-pipe</a> | <a href="https://deepwiki.com/dav0dea/goofi-pipe">AI-generated Documentation</a></h3>

# Installation
If you only want to run goofi-pipe and not edit any of the code, make sure you activated the desired Python environment with Python >=3.9 and <3.13 and run the following commands in your terminal:
```bash
pip install goofi # install goofi-pipe
goofi-pipe # start the application
```

If you want to use PyTorch-based nodes and require a specific version of CUDA, refer to the [PyTorch installation guide](https://pytorch.org/get-started/locally/) to install the correct version of PyTorch.

> [!NOTE]
> On some platforms (specifically Linux and Mac) it might be necessary to install the `liblsl` package for some of goofi-pipe's features (everything related to LSL streams).
> Follow the instructions provided [here](https://github.com/sccn/liblsl?tab=readme-ov-file#getting-and-using-liblsl), or simply install it via
> ```bash
> conda install -c conda-forge liblsl
> ```

## Development
Follow these steps if you want to adapt the code of existing nodes, or create custom new nodes. In your terminal, make sure you activated the desired Python environment with Python >=3.9 and <3.13, and that you are in the directory where you want to install goofi-pipe. Then, run the following commands:
```bash
git clone git@github.com:dav0dea/goofi-pipe.git # download the repository
cd goofi-pipe # navigate into the repository
pip install -e . # install goofi-pipe in development mode
goofi-pipe # start the application to make sure the installation was successful
```

# Basic Usage

## Accessing the Node Menu

<p align="center">
<img src="https://github.com/dav0dea/goofi-pipe/assets/49297774/358a897f-3947-495e-849a-e6d7ebce2238" width="small">
</p>

To access the node menu, simply double-click anywhere within the application window or press the 'Tab' key. The node menu allows you to add various functionalities to your pipeline. Nodes are categorized for easy access, but if you're looking for something specific, the search bar at the top is a handy tool.

## Common Parameters and Metadata

<p align="center">
<img src="https://github.com/dav0dea/goofi-pipe/assets/49297774/23ba6df7-7f28-4505-acff-205e42e48dcb" alt="Common Parameters" width="small">
</p>

**Common Parameters**: All nodes within goofi have a set of common parameters. These settings consistently dictate how the node operates within the pipeline.

- **AutoTrigger**: This option, when enabled, allows the node to be triggered automatically. When disabled,
the node is triggered when it receives input.
  
- **Max_Frequency**: This denotes the maximum rate at which computations are set for the node.

<p align="center">
<img src="https://github.com/dav0dea/goofi-pipe/assets/49297774/54604cfb-6611-4ce8-92b2-0b353584c5f5" alt="Metadata" width="small">
</p>

**Metadata**: This section conveys essential information passed between nodes. Each output node will be accompanied by its metadata, providing clarity and consistency throughout the workflow.

Here are some conventional components present in the metadata

- **Channel Dictionary**: A conventional representation of EEG channels names.
  
- **Sampling Frequency**: The rate at which data samples are measured. It's crucial for maintaining consistent data input and output across various nodes.

- **Shape of the Output**: Details the format and structure of the node's output.


## Playing with Pre-recorded EEG Signal using LSLClient

<p align="center">
<img src="https://github.com/dav0dea/goofi-pipe/assets/49297774/db340bd9-07af-470e-a791-f3c2dcf4935e" width="small">
</p>

This image showcases the process of utilizing a pre-recorded EEG signal through the `LSLClient` node. It's crucial to ensure that the `Stream Name` in the `LSLClient` node matches the stream name in the node receiving the data. This ensures data integrity and accurate signal processing in real-time.

# Patch examples

## Basic Signal Processing Patch

<p align="center">
<img src="https://github.com/dav0dea/goofi-pipe/assets/49297774/52f85dd4-6395-4eb2-a347-6cf489d659da" width="medium">
</p>

This patch provides a demonstration of basic EEG signal processing using goofi-pipe.

1. **EEGRecording**: This is the starting point where the EEG data originates.

2. **LSLClient**: The `LSLClient` node retrieves the EEG data from `EEGRecording`. Here, the visual representation of the EEG data being streamed in real-time is depicted. By default, the multiple lines in the plot correspond to the different EEG channels.

3. **Buffer**: This node holds the buffered EEG data.

4. **Psd**: Power Spectral Density (PSD) is a technique to measure a signal's power content versus frequency. In this node, the raw EEG data is transformed to exhibit its power distribution across distinct frequency bands.

5. **Math**: This node is employed to execute mathematical operations on the data. In this context, it's rescaling the values to ensure a harmonious dynamic range between 0 and 1, which is ideal for image representation. The resultant data is then visualized as an image.

One of the user-friendly features of goofi-pipe is the capability to toggle between different visualizations. By 'Ctrl+clicking' on any plot within a node, you can effortlessly switch between a line plot and an image representation, offering flexibility in data analysis.

## Sending Power Bands via Open Sound Control (OSC)

<p align="center">
<img src="https://github.com/dav0dea/goofi-pipe/assets/49297774/97576017-a737-47b9-aac6-bd0d00e0e7e9" width="medium">
</p>

Expanding on the basic patch, the advanced additions include:

- **Select**: Chooses specific EEG channels.
- **PowerBandEEG**: Computes EEG signal power across various frequency bands.
- **ExtendedTable**: Prepares data for transmission in a structured format.
- **OscOut**: Sends data using the Open-Sound-Control (OSC) protocol.

These nodes elevate data processing and communication capabilities.

## Real-Time Connectivity and Spectrogram

<p align="center">
<img src="https://github.com/dav0dea/goofi-pipe/assets/49297774/7c63a869-d20a-4f41-99fe-eb0931cebdc9" width="medium">
</p>

This patch highlights:

- **Connectivity**: Analyzes relationships between EEG channels, offering selectable methods like `wPLI`, `coherence`, `PLI`, and more.

- **Spectrogram**: Created using the `PSD` node followed by a `Buffer`, it provides a time-resolved view of the EEG signal's frequency content.

## Principal Component Analysis (PCA)
![PCA](https://github.com/dav0dea/goofi-pipe/assets/36135990/d239eed8-4552-4256-9caf-d7c2fbb937e9)

Using PCA (Principal Component Analysis) allows us to reduce the dimensionality of raw EEG data, while retaining most of the variance. We use the first three components and visualize their trajectory, allowing us to identify patterns in the data over time. The topographical maps show the contribution of each channel to the first four principal components (PCs).

## Realtime Classification

Leveraging the multimodal framework of goofi, state-of-the-art machine learning classifiers can be built on-the-fly to predict behavior from an array of different sources. Here's a brief walkthrough of three distinct examples:

### 1. EEG Signal Classification
![EEG Signal Classification](https://github.com/dav0dea/goofi-pipe/assets/36135990/2da6b555-9f79-40c7-9bd8-1f863dcf4137)
This patch captures raw EEG signals using the `EEGRecording` and `LSLClient` modules. The classifier captures data from different states indicated by the user from *n* features, which in the present case are the 64 EEG channels. Some classifiers allow for visualization of feature importance. Here we show a topomap of the distribution of feature importances on the scalp. The classifier outputs probability of being in each of the states in the training data. This prediction is smoothed using a buffer for less jittery results.
![Classifier parameters](https://github.com/dav0dea/goofi-pipe/assets/49297774/da2a86e3-efc8-4088-8d52-fb8c528dfb87)

### 2. Audio Input Classification
![Audio Input Classification](https://github.com/dav0dea/goofi-pipe/assets/49297774/4e50b13e-185d-414e-a39d-f6d39dc3e57f)
The audio input stream captures real-time sound data, which can also be passed through a classifier. Different sonic states can be predicted in realtime.

### 3. Video Input Classification
![Video Input Classification](https://github.com/dav0dea/goofi-pipe/assets/49297774/e7988ae9-cd2c-4b9f-907a-f438fd52328b)
![image_classification2](https://github.com/dav0dea/goofi-pipe/assets/49297774/77d33f2e-014f-4e3b-99fb-179f4bca1db0)
In this example, video frames are extracted using the `VideoStream` module. Similarly, prediction of labelled visual states can be achieved in realtime.
The images show how two states (being on the left or the right side of the image) can be detected using classification

These patches demonstrate the versatility of our framework in handling various types of real-time data streams for classification tasks.

## Musical Features using Biotuner

<p align="center">
<img src="https://github.com/dav0dea/goofi-pipe/assets/49297774/b426ce44-bf23-4b88-a772-5d183dc36a93" width="medium">
</p>

This patch presents a pipeline for processing EEG data to extract musical features:

- Data flows from the EEG recording through several preprocessing nodes and culminates in the **Biotuner** node, which specializes in deriving musical attributes from the EEG.

- **Biotuner** Node: With its sophisticated algorithms, Biotuner pinpoints harmonic relationships, tension, peaks, and more, essential for music theory analysis.

<p align="center">
<img src="https://github.com/dav0dea/goofi-pipe/assets/49297774/042692ae-a558-48f2-9693-d09e33240373" width="medium">
</p>

Delving into the parameters of the Biotuner node:

- `N Peaks`: The number of spectral peaks to consider.
- `F Min` & `F Max`: Defines the frequency range for analysis.
- `Precision`: Sets the precision in Hz for peak extraction.
- `Peaks Function`: Method to compute the peaks, like EMD, fixed band, or harmonic recurrence.
- `N Harm Subharm` & `N Harm Extended`: Configures number of harmonics used in different computations.
- `Delta Lim`: Defines the maximal distance between two subharmonics to include in subharmonic tension computation.

For a deeper understanding and advanced configurations, consult the [Biotuner repository](https://github.com/AntoineBellemare/biotuner).


# Data Types

To simplify understanding, we've associated specific shapes with data types at the inputs and outputs of nodes:

- **Circles**: Represent arrays.
- **Triangles**: Represent strings.
- **Squares**: Represent tables.


# Node Categories

<!-- AUTO-GENERATED NODE LIST -->
<!-- !!GOOFI_PIPE_NODE_LIST_START!! -->
## Analysis

Nodes that perform analysis on the data.

<details><summary>View Nodes</summary>

<details><summary>&emsp;Audio2Txt</summary>

## Audio2Txt
```
Inputs:
  - prompt: InputSlot(dtype=<DataType.STRING: 1>, trigger_process=False, data=None)
  - audio: ARRAY

Outputs:
  - generated_text: STRING
```

This node converts an audio signal into natural language text by using a large language model specialized for audio captioning or transcription. It can optionally take a prompt to guide the generation of the output text. The audio is processed and sent to the selected provider, which returns a generated text description or transcription of the audio content.

### Inputs
- prompt: Optional text prompt that can guide or condition the text generation.
- audio: Array containing the audio data to be transcribed or described.

### Outputs
- generated_text: The text generated from the input audio, which may be a transcription or caption depending on the model and prompt.

  </details>

<details><summary>&emsp;AudioTagging</summary>

## AudioTagging
```
Inputs:
  - audioIn: ARRAY

Outputs:
  - tags: STRING
  - probabilities: ARRAY
  - embedding: ARRAY
```

AudioTagging node performs automatic audio tagging and embedding extraction from input audio data. It uses a pre-trained audio tagging model to identify the most likely audio classes present in the input, returning their probabilities and associated feature embeddings. The output tags are filtered by confidence and optionally by category.

### Inputs
- audioIn: 1D audio data array sampled at 32 kHz.

### Outputs
- tags: List of detected audio tag names as a newline-separated string.
- probabilities: Array of confidence values for each detected tag.
- embedding: Feature embedding array representing the input audio.

  </details>

<details><summary>&emsp;Avalanches</summary>

## Avalanches
```
Inputs:
  - data: ARRAY

Outputs:
  - size: ARRAY
  - duration: ARRAY
```

This node detects neuronal avalanches in the input signal using the EdgeOfPy library. Avalanches are cascades of activity within the signal, and the node outputs the size and duration of each detected avalanche event.

### Inputs
- data: Input array of signals, where each row represents a separate channel and each column is a time point.

### Outputs
- size: Array containing the size (number of events) of each detected avalanche.
- duration: Array containing the duration (in seconds) of each detected avalanche.

  </details>

<details><summary>&emsp;Binarize</summary>

## Binarize
```
Inputs:
  - data: ARRAY

Outputs:
  - bin_data: ARRAY
```

This node transforms an input array into a binary array by applying a thresholding operation. For each value in the input, it assigns a 1 or 0 based on whether the value meets specified threshold criteria, effectively creating a binary representation of the original data.

### Inputs
- data: Array data to be binarized.

### Outputs
- bin_data: The binarized version of the input array, with each value set to either 1 or 0 based on the thresholding operation.

  </details>

<details><summary>&emsp;Bioelements</summary>

## Bioelements
```
Inputs:
  - data: ARRAY

Outputs:
  - elements: TABLE
```

This node identifies chemical elements present in an input 1D frequency array by matching its spectral features against a database of known air element spectral lines. It extracts the most frequently matched elements and provides their names, spectral regions, and types as output.

### Inputs
- data: A 1D array representing signal frequencies to be analyzed for spectral characteristics.

### Outputs
- elements: A table containing the names of detected elements, their corresponding spectral regions, and their types, based on the matches found in the spectral database.

  </details>

<details><summary>&emsp;Bioplanets</summary>

## Bioplanets
```
Inputs:
  - peaks: ARRAY

Outputs:
  - planets: TABLE
  - top_planets: STRING
```

This node analyzes an array of peak frequencies and identifies which of the main solar system planets (Venus, Earth, Mars, Jupiter, Saturn) have spectral lines near those peaks, based on a reference planetary spectrum dataset. For each planet, it outputs the list of matched wavelengths, and also provides a summary indicating which planets have the most matched peaks.

### Inputs
- peaks: A 1D array of detected frequency peaks to be analyzed.

### Outputs
- planets: A table containing, for each planet, the array of matched wavelengths corresponding to the input peaks.
- top_planets: A string listing the planets with the highest number of matched spectral peaks, ranked in order.

  </details>

<details><summary>&emsp;Biorhythms</summary>

## Biorhythms
```
Inputs:
  - tuning: ARRAY

Outputs:
  - pulses: ARRAY
  - steps: ARRAY
  - offsets: ARRAY
```

This node analyzes an input tuning array and generates rhythm patterns by deriving Euclidean-like, consonant rhythmic groupings from the harmonic relationships in the tuning. It outputs the number of pulses, the total number of steps, and the phase offsets for each rhythm calculated from the input tuning.

### Inputs
- tuning: A 1D array representing a set of frequency or pitch values to be used for harmonic analysis and rhythm derivation.

### Outputs
- pulses: An array indicating the number of pulses for each generated rhythm pattern.
- steps: An array indicating the number of steps for each generated rhythm pattern.
- offsets: An array indicating the phase offset for each generated rhythm pattern.

  </details>

<details><summary>&emsp;Biotuner</summary>

## Biotuner
```
Inputs:
  - data: ARRAY

Outputs:
  - harmsim: ARRAY
  - tenney: ARRAY
  - subharm_tension: ARRAY
  - cons: ARRAY
  - peaks_ratios_tuning: ARRAY
  - harm_tuning: ARRAY
  - peaks: ARRAY
  - amps: ARRAY
  - extended_peaks: ARRAY
  - extended_amps: ARRAY
```

This node extracts dominant frequency peaks from an incoming 1D signal and computes several music-theoretical and harmonic metrics from the spectral content using the Biotuner library. It provides both raw peak data and a variety of measures relating to harmonicity and consonance, potentially useful for music analysis, neurofeedback, or real-time audio investigations.

### Inputs
- data: A 1D array containing the signal from which spectral peaks and associated metrics will be derived.

### Outputs
- harmsim: Harmonic similarity metric for the extracted peaks.
- tenney: Tenney height metric, a consonance measure, for the peaks.
- subharm_tension: Subharmonic tension metric, indicating harmonic tension based on subharmonics.
- cons: General consonance value computed from the peaks.
- peaks_ratios_tuning: Ratios between the detected peaks, reflecting interval relationships for tuning.
- harm_tuning: Harmonic tuning information for the peaks.
- peaks: Array of main frequency peaks detected in the input signal.
- amps: Amplitudes corresponding to the detected main frequency peaks.
- extended_peaks: Additional peaks derived by harmonic extension methods.
- extended_amps: Amplitudes of the extended peaks.

  </details>

<details><summary>&emsp;CardiacRespiration</summary>

## CardiacRespiration
```
Inputs:
  - data: ARRAY

Outputs:
  - cardiac: ARRAY
```

This node extracts a respiration signal derived from cardiac activity in a 1D physiological waveform. It processes either ECG or (in the future) PPG signals and returns an estimated respiratory waveform (EDR) based on cardiac information.

### Inputs
- data: 1D array containing a physiological waveform (such as ECG), including associated metadata like sampling rate.

### Outputs
- cardiac: 1D array containing the cardiac-derived respiration (EDR) signal with original metadata.

  </details>

<details><summary>&emsp;CardioRespiratoryVariability</summary>

## CardioRespiratoryVariability
```
Inputs:
  - data: ARRAY

Outputs:
  - Mean: ARRAY
  - SDNN: ARRAY
  - SDSD: ARRAY
  - RMSSD: ARRAY
  - VLF: ARRAY
  - LF: ARRAY
  - HF: ARRAY
  - LF/HF: ARRAY
  - Peaks: ARRAY
  - Rate: ARRAY
```

This node computes a range of cardiorespiratory variability metrics from an input 1D signal, such as photoplethysmogram (PPG), electrocardiogram (ECG), or respiratory (RSP) data. It processes the signal to detect cycles (e.g., heartbeats or breaths), extracts rate information, and calculates time-domain and frequency-domain variability features relevant for heart rate variability (HRV) or respiratory rate variability (RRV) analysis.

### Inputs
- data: A 1D array containing the physiological signal to be analyzed (e.g., PPG, ECG, or RSP), along with associated sampling rate metadata.

### Outputs
- Mean: Mean interval between cycles (NN or BB intervals, depending on signal type).
- SDNN: Standard deviation of NN or BB intervals, a measure of overall variability.
- SDSD: Standard deviation of successive differences between intervals.
- RMSSD: Root mean square of successive differences between intervals.
- VLF: Power in the very low frequency band.
- LF: Power in the low frequency band.
- HF: Power in the high frequency band.
- LF/HF: Ratio of low frequency to high frequency power.
- Peaks: Indices where peaks (e.g., heartbeats or breaths) are detected within the input signal.
- Rate: Instantaneous rate (e.g., heart or respiratory rate) for each detected cycle.

  </details>

<details><summary>&emsp;Classifier</summary>

## Classifier
```
Inputs:
  - data: ARRAY

Outputs:
  - probs: ARRAY
  - feature_importances: ARRAY
```

This node performs supervised classification on streaming data using a selection of machine learning algorithms, such as Naive Bayes, SVM, Random Forest, Logistic Regression, or K-Nearest Neighbors. The node allows real-time training, prediction, saving/loading of the training set, and outputs both class probabilities and feature importances (when available) for the current input. It is used to infer and monitor discrete classes or states from continuous or multi-dimensional input data.

### Inputs
- data: Multidimensional numerical array to be classified, typically shaped as (n_features,) or (n_features, n_samples).

### Outputs
- probs: Class probabilities for the input data, along with metadata such as the chosen classifier and training set size per class.
- feature_importances: Numeric array of feature importances or weights for the classification model, when applicable. If not available for the chosen classifier, this output is None.

  </details>

<details><summary>&emsp;Clustering</summary>

## Clustering
```
Inputs:
  - matrix: ARRAY

Outputs:
  - cluster_labels: ARRAY
  - cluster_centers: ARRAY
```

This node performs clustering on input data using either the KMeans or Agglomerative Clustering algorithms. It assigns each data point in the input matrix to a cluster, and for KMeans, also computes the centers of each cluster.

### Inputs
- matrix: An array-like structure containing the data to be clustered. Each row should represent a sample, and each column a feature.

### Outputs
- cluster_labels: An array with a cluster label for each input sample, indicating its assigned cluster.
- cluster_centers: An array of the centers of the clusters (only produced when using the KMeans algorithm).

  </details>

<details><summary>&emsp;Compass</summary>

## Compass
```
Inputs:
  - pole1: ARRAY
  - pole2: ARRAY

Outputs:
  - angles: ARRAY
```

Computes the angular differences between two points in N-dimensional space. The node receives two N-dimensional vectors and calculates the sequence of angles between corresponding vector elements, expressing the direction from the first input vector (pole1) to the second (pole2) projected onto each pair of adjacent dimensions. The angles are given in degrees and normalized to the range [0, 360).

### Inputs
- pole1: First N-dimensional array representing the initial point.
- pole2: Second N-dimensional array representing the target point.

### Outputs
- angles: Array of N-1 angles (in degrees) representing the orientation difference between pole1 and pole2 in each pair of adjacent dimensions.

  </details>

<details><summary>&emsp;Connectivity</summary>

## Connectivity
```
Inputs:
  - data: ARRAY

Outputs:
  - matrix: ARRAY
```

This node computes connectivity matrices from multichannel 2D signals. It supports both classical and biotuner-based methods for estimating pairwise relationships between channels, such as coherence, wPLI, PLV, covariance, or various harmonic similarity measures. Optionally, the output matrix can be binarized based on a threshold.

### Inputs
- data: 2D array signal data, typically with shape (channels, samples), and accompanying metadata.

### Outputs
- matrix: 2D array representing the computed connectivity between channels, with original metadata attached.

  </details>

<details><summary>&emsp;Coord2loc</summary>

## Coord2loc
```
Inputs:
  - latitude: ARRAY
  - longitude: ARRAY

Outputs:
  - coord_info: TABLE
  - water_situation: ARRAY
```

This node converts geographic coordinates (latitude and longitude) into human-readable location information using reverse geocoding. It determines key location details such as city, state, country, road, and village, and provides both a structured location summary and an indication of whether the coordinates represent a valid land location or are in the ocean.

### Inputs
- latitude: The latitude coordinate to be reverse geocoded.
- longitude: The longitude coordinate to be reverse geocoded.

### Outputs
- coord_info: A table containing information about the location corresponding to the input coordinates, such as city, state, country, road, village, and the full address. If the coordinates are invalid or not associated with a land address, it returns a message indicating the location cannot be determined.
- water_situation: An array indicating whether the coordinates correspond to a recognized land location (0) or a water/non-land location (1).

  </details>

<details><summary>&emsp;Correlation</summary>

## Correlation
```
Inputs:
  - data1: ARRAY
  - data2: ARRAY

Outputs:
  - pearson: ARRAY
  - pval: ARRAY
```

Calculates the Pearson correlation coefficient and p-value between two input arrays. This node measures the linear correlation between the inputs and returns both the correlation values and their statistical significance.

### Inputs
- data1: First array of data to correlate.
- data2: Second array of data to correlate, broadcasted to the same shape as data1 if necessary.

### Outputs
- pearson: Array of Pearson correlation coefficients between data1 and data2.
- pval: Array of p-values indicating the statistical significance of the correlation.

  </details>

<details><summary>&emsp;DimensionalityReduction</summary>

## DimensionalityReduction
```
Inputs:
  - data: ARRAY
  - new_data: ARRAY

Outputs:
  - transformed: ARRAY
  - new_components: ARRAY
```

Performs dimensionality reduction on array data using one of several algorithms (PCA, t-SNE, or UMAP), reducing high-dimensional input data to a lower-dimensional representation. This node can also optionally transform new incoming data samples into the previously computed low-dimensional space, when supported by the selected algorithm.

### Inputs
- data: The original array data to be reduced in dimensionality. Must be 2D.
- new_data: New array data samples to be projected into the computed lower-dimensional space using the fitted model.

### Outputs
- transformed: The array data transformed into the lower-dimensional space, along with updated metadata.
- new_components: The new data samples transformed into the same lower-dimensional space, with updated metadata. Only provided if new_data is given and supported by the selected method.

  </details>

<details><summary>&emsp;DissonanceCurve</summary>

## DissonanceCurve
```
Inputs:
  - peaks: ARRAY
  - amps: ARRAY

Outputs:
  - dissonance_curve: ARRAY
  - tuning: ARRAY
  - avg_dissonance: ARRAY
```

This node calculates a dissonance curve for a set of spectral peaks and their amplitudes, providing a quantitative measure of sensory dissonance as a function of interval ratios. It is typically used to analyze the dissonance of complex tones and suggest tunings that minimize dissonance.

### Inputs
- peaks: An array of frequency peaks (typically from a spectrum).
- amps: An array of amplitudes corresponding to the frequency peaks.

### Outputs
- dissonance_curve: An array representing the computed dissonance curve over a range of interval ratios.
- tuning: An array of interval ratios (relative tunings) corresponding to the dissonance curve.
- avg_dissonance: An array containing the average dissonance value across tested intervals.

  </details>

<details><summary>&emsp;DreamInceptor</summary>

## DreamInceptor
```
Inputs:
  - data: ARRAY
  - start: ARRAY
  - reset: ARRAY

Outputs:
  - trigger: ARRAY
  - z_theta_alpha: ARRAY
  - z_lempel_ziv: ARRAY
  - baseline_stats: TABLE
  - hypnodensities: ARRAY
```

This node monitors an incoming EEG signal and detects specific brain activity patterns suitable for targeted dream intervention (dream inception). Depending on configuration, it operates in two modes: (1) theta/alpha z-score detection, using baseline statistics to monitor significant changes in spectral ratios or signal complexity; or (2) hypnodensity-based detection, leveraging a deep learning model to compute state probabilities and entropy for distinguishing sleep stages. When detection criteria are met, the node outputs a trigger signal indicating the optimal moment for dream incubation, along with relevant feature data for tracking and analysis.

### Inputs
- data: One-dimensional EEG time series data used for ongoing analysis and detection.
- start: Optional control input to initiate the dream inception detection process.
- reset: Optional control input to abort and reset the detection process and baseline.

### Outputs
- trigger: Array output (single value) set to 1 when detection criteria are met (indicating to trigger dream incubation), or 0 when baseline collection is complete, otherwise None.
- z_theta_alpha: Array output with the current z-scored theta/alpha ratio for the EEG segment (only in theta_alpha mode).
- z_lempel_ziv: Array output with the current z-scored Lempel-Ziv complexity for the EEG segment (only in theta_alpha mode).
- baseline_stats: Table of mean or quantile baseline statistics for theta/alpha and Lempel-Ziv values (provided during and after baseline collection; only available in theta_alpha mode).
- hypnodensities: Array output with 6 values per window: 5-element hypnodensity state probabilities (from a neural network classifier) plus normalized entropy, updated each window (only in hypnodensity mode).

  </details>

<details><summary>&emsp;EEGEmbedding</summary>

## EEGEmbedding
```
Inputs:
  - eeg: ARRAY

Outputs:
  - embeddings: ARRAY
```

The EEGEmbedding node extracts deep neural embeddings from multi-channel EEG data using a pre-trained EEG encoder model. It takes raw EEG signals, processes them through a neural network to generate feature embeddings that represent the temporal and spatial information in the brain activity. This is useful for downstream tasks such as brain decoding, classification, or similarity analysis.

### Inputs
- eeg: EEG array data with 128 channels, at least 440 samples, and a sampling frequency of 1000 Hz.

### Outputs
- embeddings: Neural feature embeddings extracted from the input EEG data as a NumPy array.

  </details>

<details><summary>&emsp;EigenDecomposition</summary>

## EigenDecomposition
```
Inputs:
  - matrix: ARRAY

Outputs:
  - eigenvalues: ARRAY
  - eigenvectors: ARRAY
```

Performs eigen decomposition on a 2D matrix input. Optionally, the node can compute the (unnormalized or normalized) Laplacian of the matrix before decomposition. Different algorithms for eigenvalue and eigenvector computation are supported. The node returns both the eigenvalues and eigenvectors of the (transformed) matrix, with outputs ordered as specified and, if needed, with a consistent sign orientation.

### Inputs
- matrix: A 2D array representing the matrix to decompose.

### Outputs
- eigenvalues: Array containing the eigenvalues of the input (or Laplacian) matrix.
- eigenvectors: 2D array where each column is an eigenvector corresponding to an eigenvalue.

  </details>

<details><summary>&emsp;Embedding</summary>

## Embedding
```
Inputs:
  - text: STRING
  - data: ARRAY

Outputs:
  - text_embeddings: ARRAY
  - data_embeddings: ARRAY
```

This node generates embeddings for text and image data using a selectable machine learning model. It takes either text, array data (such as an image), or both, and outputs their respective embeddings as arrays. Different models are used depending on the selected embedding method, supporting both text embeddings (via models like CLIP, SBERT, FastText, Word2Vec) and image embeddings (via CLIP). The node processes the input, computes fixed-size vector representations (embeddings), and outputs them for downstream use.

### Inputs
- text: Input string to be converted into an embedding vector or set of vectors.
- data: Input array data, typically representing an image, to be converted into an embedding vector.

### Outputs
- text_embeddings: Array containing the embedding(s) generated from the input text.
- data_embeddings: Array containing the embedding generated from the input array data.

  </details>

<details><summary>&emsp;ERP</summary>

## ERP
```
Inputs:
  - signal: ARRAY
  - trigger: ARRAY

Outputs:
  - erp: ARRAY
```

This node computes the event-related potential (ERP) by averaging segments of input signals that are time-locked to external triggers. Each time a trigger is received, it collects a segment of the signal of fixed duration following that trigger and averages these segments in real time to update the ERP. The result is an averaged response to repeated events or stimuli, useful in signal processing applications such as EEG analysis.

### Inputs
- signal: Incoming array data representing the continuous time-series signal to be analyzed and averaged.
- trigger: Array indicating the presence of a trigger event, used to segment the signal for ERP extraction.

### Outputs
- erp: The current averaged event-related potential as an array, along with updated metadata.

  </details>

<details><summary>&emsp;FacialExpression</summary>

## FacialExpression
```
Inputs:
  - image: ARRAY

Outputs:
  - emotion_probabilities: ARRAY
  - action_units: ARRAY
  - main_emotion: STRING
```

This node analyzes an input image and detects human facial expressions and action units using the py-feat library. It identifies the probabilities for several emotions, extracts facial action units, and determines the main emotion detected in the face within the image.

### Inputs
- image: An image (numpy array) containing a face to be analyzed.

### Outputs
- emotion_probabilities: An array containing the detection probabilities for each emotion (anger, disgust, fear, happiness, sadness, surprise, neutral).
- action_units: An array representing the detected facial muscle action units (AUs) based on the given face.
- main_emotion: The label of the primary emotion detected in the face.

  </details>

<details><summary>&emsp;Fractality</summary>

## Fractality
```
Inputs:
  - data_input: ARRAY

Outputs:
  - fractal_dimension: ARRAY
```

This node computes the fractal dimension or fractal characteristics of input data using a variety of methods, including several signal-based estimators (such as Hurst and Higuchi) and a box counting method for 2D arrays. It is applicable to both time series and image data, depending on the selected method.

### Inputs
- data_input: An array representing either a time series (1D or 2D) or a 2D image (grayscale or RGB).

### Outputs
- fractal_dimension: The calculated fractal dimension or feature value based on the chosen method.

  </details>

<details><summary>&emsp;GraphMetrics</summary>

## GraphMetrics
```
Inputs:
  - matrix: ARRAY

Outputs:
  - clustering_coefficient: ARRAY
  - characteristic_path_length: ARRAY
  - betweenness_centrality: ARRAY
  - degree_centrality: ARRAY
  - assortativity: ARRAY
  - transitivity: ARRAY
```

This node computes several important graph-theoretical metrics from a given adjacency matrix representing an undirected graph. The node analyzes the connectivity and topological features of the input network, returning quantitative measures that describe properties such as node centrality, clustering, assortativity, and overall network structure.

### Inputs
- matrix: A 2D symmetric adjacency matrix representing an undirected graph.

### Outputs
- clustering_coefficient: The average clustering coefficient of the graph, indicating the tendency of nodes to form clusters.
- characteristic_path_length: The average shortest path length between all pairs of nodes in the graph.
- betweenness_centrality: The betweenness centrality for each node, measuring the extent to which a node lies on shortest paths between other nodes.
- degree_centrality: The degree centrality for each node, indicating how many connections each node has relative to the rest of the graph.
- assortativity: The degree assortativity coefficient, representing the similarity of connections in the graph with respect to node degree.
- transitivity: The transitivity (global clustering coefficient) of the graph, measuring the overall probability that the adjacent nodes of a node are connected.

  </details>

<details><summary>&emsp;HarmonicSpectrum</summary>

## HarmonicSpectrum
```
Inputs:
  - psd: ARRAY

Outputs:
  - harmonic_spectrum: ARRAY
  - max_harmonicity: ARRAY
  - avg_harmonicity: ARRAY
```

This node computes a harmonic spectrum of an input power spectral density (PSD) array by analyzing the harmonic relationships between frequency components. It calculates harmonicity metrics that quantify how harmonically related different spectral components are, outputting both the complete harmonic spectrum as well as summary statistics indicating the maximum and average harmonicity across frequencies.

### Inputs
- psd: An array representing the power spectral density of the signal, with frequency information included in its metadata.

### Outputs
- harmonic_spectrum: An array representing the harmonicity value at each frequency bin, indicating the degree to which each frequency is harmonically related to the others in the spectrum.
- max_harmonicity: The maximum harmonicity value found across all frequency bins, summarizing the most harmonically related component in the spectrum.
- avg_harmonicity: The average harmonicity value across all frequency bins, summarizing the overall harmonicity of the spectrum.

  </details>

<details><summary>&emsp;Img2Txt</summary>

## Img2Txt
```
Inputs:
  - image: ARRAY

Outputs:
  - generated_text: STRING
```

This node takes an input image and generates a textual description or caption for it using a selectable large language model with vision capability (such as Huggingface Llama, OpenAI GPT, or Ollama models). The image is automatically preprocessed and sent to the selected model along with a user-defined prompt. The output is the text generated by the model that describes the content of the image.

### Inputs
- image: An array representing the input image data to be captioned.

### Outputs
- generated_text: The textual description or caption generated by the selected model for the input image.

  </details>

<details><summary>&emsp;LempelZiv</summary>

## LempelZiv
```
Inputs:
  - data: ARRAY

Outputs:
  - lzc: ARRAY
```

This node computes the Lempel-Ziv complexity (LZC) of input array data. The LZC is a measure of complexity in a binary sequence, often used in signal analysis to quantify regularity or randomness. The input array is binarized using either a mean or median threshold along a specified axis, and then the Lempel-Ziv complexity is computed for each segment. The output is an array of LZC values, retaining the input metadata.

### Inputs
- data: Input array data to be analyzed.

### Outputs
- lzc: Array containing the Lempel-Ziv complexity values calculated from the input data, with metadata attached.

  </details>

<details><summary>&emsp;Monolith</summary>

## Monolith
```
Inputs:
  - data: ARRAY

Outputs:
  - features: ARRAY
  - clean_data: ARRAY
```

This node preprocesses incoming multichannel array data and extracts a broad set of signal features. It applies standard signal preprocessing (bandpass and notch filtering, DC offset removal, clipping, standardization), then computes a variety of channel-wise and non-channel-wise feature descriptors to summarize the data. The node outputs both the extracted features and the cleaned, preprocessed signal.

### Inputs
- data: Multichannel array data (e.g., time series data) with associated metadata.

### Outputs
- features: Extracted features summarizing the input signal, along with updated metadata.
- clean_data: The preprocessed (filtered, clipped, standardized) signal data, with unchanged metadata.

  </details>

<details><summary>&emsp;PCA</summary>

## PCA
```
Inputs:
  - data: ARRAY

Outputs:
  - principal_components: ARRAY
```

This node performs Principal Component Analysis (PCA) on 2D array input data. It extracts a specified number of principal components, which are orthogonal vectors that capture the directions of maximum variance in the data. The node outputs these principal components as a matrix.

### Inputs
- data: 2D array to analyze, where each row is a sample and each column is a feature.

### Outputs
- principal_components: Array of the computed principal component vectors, along with associated metadata.

  </details>

<details><summary>&emsp;PhiID</summary>

## PhiID
```
Inputs:
  - matrix: ARRAY

Outputs:
  - PhiID: ARRAY
  - inf_dyn: ARRAY
  - IIT: ARRAY
```

This node computes Partial Information Decomposition (PhiID) metrics between pairs of signals or between one signal and the others in a multichannel dataset. It uses the phyid package to estimate fine-grained informational components that describe unique, redundant, synergistic, and transfer relationships between signals over a specified time lag. The node outputs the full set of PhiID "atom" values, as well as summary metrics relevant to information dynamics and integrated information theory.

### Inputs
- matrix: A 2D array (channels x timepoints) representing multichannel time series data.

### Outputs
- PhiID: An array containing the values of all atomic PhiID terms for each channel pair or one-vs-others, with metadata specifying the channel axes.
- inf_dyn: An array of summary metrics relevant to information dynamics, for each channel pair or one-vs-others, with corresponding labels.
- IIT: An array of summary metrics corresponding to integrated information theory concepts, for each channel pair or one-vs-others, with appropriate metadata.

  </details>

<details><summary>&emsp;PoseEstimation</summary>

## PoseEstimation
```
Inputs:
  - image: ARRAY

Outputs:
  - pose: ARRAY
```

Estimates the 3D positions of hand landmarks from an input image using a machine learning model. The node processes images to detect a single hand and outputs the detected hand pose as a set of 3D landmark coordinates along with related metadata.

### Inputs
- image: A 2D or 3D array representing the input image for hand pose estimation.

### Outputs
- pose: The detected 3D hand landmarks as an array transposed to (3, 21), with accompanying metadata including the hand handedness and landmark channel names.

  </details>

<details><summary>&emsp;PowerBand</summary>

## PowerBand
```
Inputs:
  - data: ARRAY

Outputs:
  - power: ARRAY
```

This node calculates the total power within a specified frequency band from input data representing a power spectral density (PSD). It sums the power values within a chosen frequency range to output either the absolute or relative band power. The output preserves the input's metadata.

### Inputs
- data: An array containing PSD values with associated frequency information in the metadata.

### Outputs
- power: The computed band power as an array, along with the original metadata.

  </details>

<details><summary>&emsp;PowerBandEEG</summary>

## PowerBandEEG
```
Inputs:
  - data: ARRAY

Outputs:
  - delta: ARRAY
  - theta: ARRAY
  - alpha: ARRAY
  - lowbeta: ARRAY
  - highbeta: ARRAY
  - gamma: ARRAY
```

This node calculates the power in standard EEG frequency bands (delta, theta, alpha, low beta, high beta, and gamma) from an input power spectral density (PSD) array. It processes either 1D or 2D PSD data and outputs the total (or relative) power within each band as a new array. Each output also provides information about the frequency range used.

### Inputs
- data: Power spectral density (PSD) data as a 1D or 2D array, with corresponding frequency values provided in the metadata.

### Outputs
- delta: Power in the delta band (1–3 Hz), along with band metadata.
- theta: Power in the theta band (3–7 Hz), along with band metadata.
- alpha: Power in the alpha band (7–12 Hz), along with band metadata.
- lowbeta: Power in the low beta band (12–20 Hz), along with band metadata.
- highbeta: Power in the high beta band (20–30 Hz), along with band metadata.
- gamma: Power in the gamma band (30–50 Hz), along with band metadata.

  </details>

<details><summary>&emsp;ProbabilityMatrix</summary>

## ProbabilityMatrix
```
Inputs:
  - input_data: ARRAY

Outputs:
  - data: ARRAY
```

This node computes and updates a state transition probability matrix based on an incoming sequence of data values. Each data value is discretized (rounded to 2 decimal places), and the node tracks transitions between consecutive values, incrementally building a transition matrix representing the empirical probabilities of moving from one state to another within the observed sequence.

### Inputs
- input_data: Array of numerical data points, which are discretized and used to build the transition probability matrix by tracking transitions between consecutive values.

### Outputs
- data: The current transition probability matrix as a NumPy array, along with associated metadata such as sampling frequency.

  </details>

<details><summary>&emsp;SpectroMorphology</summary>

## SpectroMorphology
```
Inputs:
  - data: ARRAY

Outputs:
  - spectro: ARRAY
```

This node calculates spectromorphological features on a 1D audio signal. It analyzes the input time series and extracts a selected spectral feature over time, returning the feature values aligned with their corresponding timestamps.

### Inputs
- data: A 1D array containing the audio signal to be analyzed, along with metadata such as the sampling frequency.

### Outputs
- spectro: A tuple consisting of an array of computed spectromorphological feature values and the original metadata.

  </details>

<details><summary>&emsp;SpeechSynthesis</summary>

## SpeechSynthesis
```
Inputs:
  - text: STRING
  - voice: ARRAY

Outputs:
  - speech: ARRAY
  - transcript: STRING
```

This node provides speech synthesis and transcription capabilities using OpenAI's API. It can convert input text into synthesized speech audio, or transcribe input speech audio into text.

### Inputs
- text: The text string to be converted into speech audio.
- voice: An array representing audio data to be transcribed into text.

### Outputs
- speech: An array containing the synthesized speech audio corresponding to the input text, or an empty array if text input is not provided.
- transcript: The transcribed text from the input voice array, or an empty string if voice input is not provided.

  </details>

<details><summary>&emsp;TotoEmbedding</summary>

## TotoEmbedding
```
Inputs:
  - timeseries: ARRAY

Outputs:
  - embedding: ARRAY
```

This node generates fixed-size embeddings from timeseries data using the Toto foundation model. It processes 1D (time) or 2D (channels × time) array inputs, passing them through the Toto model and outputting the resulting embeddings. The embeddings can be optionally averaged across channels or time segments, resulting in a condensed feature representation suitable for downstream machine learning or analysis tasks.

### Inputs
- timeseries: An array representing a single-channel timeseries (1D) or multi-channel timeseries (2D, channels × time).

### Outputs
- embedding: An array containing the Toto model embedding of the input timeseries, optionally averaged across channels and/or time.

  </details>

<details><summary>&emsp;TransitionalHarmony</summary>

## TransitionalHarmony
```
Inputs:
  - data: ARRAY

Outputs:
  - trans_harm: ARRAY
  - melody: ARRAY
```

This node computes a measure of transitional harmony between two segments of an input 1D array. The input data is split in half, and dominant frequency peaks are extracted from each half using a selected method. It then compares the frequency peaks between the two halves to analyze subharmonic relationships, providing an array representing harmonic tension over time as well as the harmonic pairs used in this analysis.

### Inputs
- data: 1D array containing the input signal to be analyzed.

### Outputs
- trans_harm: Array representing the transitional harmonic tension computed between peaks extracted from the two halves of the input.
- melody: Array of harmonic peak pairs that contribute to the transitional harmonic analysis.

  </details>

<details><summary>&emsp;TuningColors</summary>

## TuningColors
```
Inputs:
  - data: ARRAY

Outputs:
  - hue: ARRAY
  - saturation: ARRAY
  - value: ARRAY
  - color_names: STRING
```

This node converts a musical scale represented as an array of frequency ratios into corresponding HSV color values, based on the frequency and averaged consonance of each step in the scale. The hue represents mapped frequency, saturation encodes consonance, and the value (brightness) is fixed. Additionally, the node provides readable color names or HEX codes for the first few colors calculated from the scale.

### Inputs
- data: 1D array representing a musical scale, where the first element is the fundamental frequency in Hz and the rest are frequency ratios.

### Outputs
- hue: Array of hue values (float), one for each note in the scale except the fundamental, representing converted pitch.
- saturation: Array of saturation values (float), encoding the consonance for each scale step.
- value: Array of value (brightness) values (float), set to a constant for each scale step.
- color_names: String containing the names or HEX codes of the first few color representations corresponding to the scale notes (excluding the fundamental).

  </details>

<details><summary>&emsp;TuningMatrix</summary>

## TuningMatrix
```
Inputs:
  - tuning: ARRAY

Outputs:
  - matrix: ARRAY
  - metric_per_step: ARRAY
  - metric: ARRAY
```

This node computes a tuning matrix and related metrics from an input tuning array using a selectable metric function. It processes the tuning data to generate a matrix that characterizes relationships or similarities between tunings, as well as metrics measuring aspects of the tuning structure.

### Inputs
- tuning: An array containing tuning data to be analyzed.

### Outputs
- matrix: The computed tuning matrix, expressing inter-relations between elements in the input tuning array. Depending on configuration, this may be a normalized matrix or may be set to None for certain ratio types.
- metric_per_step: An array containing the metric score calculated for each step or pair in the tuning.
- metric: An array containing overall metric values summarizing properties of the tuning.

  </details>

<details><summary>&emsp;TuningReduction</summary>

## TuningReduction
```
Inputs:
  - tuning: ARRAY

Outputs:
  - reduced: ARRAY
```

This node reduces a given tuning array to a simplified musical mode using one of several harmonicity or consonance methods. It processes a 1D array of tuning values and outputs a reduced set of pitches according to the selected metric.

### Inputs
- tuning: 1D array of tuning values to be reduced.

### Outputs
- reduced: Reduced array of tuning values representing the extracted musical mode.

  </details>

<details><summary>&emsp;VAMP</summary>

## VAMP
```
Inputs:
  - data: ARRAY

Outputs:
  - comps: ARRAY
```

This node computes a variational approach for Markov processes (VAMP) dimensionality reduction on incoming 2D array data. It accumulates epochs of data, fits a VAMP model using a sliding time-lagged window, and projects input data onto learned VAMP components in real-time. The node is typically used to extract meaningful low-dimensional temporal features from multivariate time series.

### Inputs
- data: 2D array input data representing multichannel time series.

### Outputs
- comps: 2D array of VAMP components for the input data, with updated metadata.

  </details>

<details><summary>&emsp;VocalExpression</summary>

## VocalExpression
```
Inputs:
  - data: ARRAY

Outputs:
  - prosody_label: STRING
  - burst_label: STRING
  - prosody_score: ARRAY
  - burst_score: ARRAY
```

This node analyzes voice audio input to detect emotional expressions using the Hume AI API. It processes a 1D audio signal and returns the predominant vocal emotions and their associated confidence scores, based on prosody and burst vocal analytics.

### Inputs
- data: A 1D array containing the raw audio waveform data.

### Outputs
- prosody_label: The name of the strongest detected emotion from prosody-based analysis.
- burst_label: The name of the strongest detected emotion from burst-based analysis.
- prosody_score: The confidence score of the detected prosody emotion as a 1-element array.
- burst_score: The confidence score of the detected burst emotion as a 1-element array.

  </details>

<details><summary>&emsp;Walker</summary>

## Walker
```
Inputs:
  - angle: ARRAY
  - velocity: ARRAY
  - water: ARRAY

Outputs:
  - latitude: ARRAY
  - longitude: ARRAY
```

Simulates a step-by-step movement of a point (walker) on the Earth's surface based on a direction (angle), speed (velocity), and whether it is moving on water (water). The node calculates the new latitude and longitude from the previous position for each input, accounting for the Earth's curvature and constraints such as pole crossings and longitude normalization.

### Inputs
- angle: The direction(s) of movement in degrees, as an array.
- velocity: The distance(s) to move per step, as an array. Might be scaled up if moving through water.
- water: An array indicating if the movement is over water (1) or not (0), which affects the speed.

### Outputs
- latitude: The updated latitude(s) after applying movement and corrections for the globe's boundaries.
- longitude: The updated longitude(s) after applying movement and corrections for the globe's boundaries.

  </details>

</details>

## Array

Nodes implementing array operations.

<details><summary>View Nodes</summary>

<details><summary>&emsp;Clip</summary>

## Clip
```
Inputs:
  - array: ARRAY

Outputs:
  - out: ARRAY
```

Clips the values of an input array so that they stay within a specified minimum and maximum range. Any values below the minimum are set to the minimum, and any values above the maximum are set to the maximum.

### Inputs
- array: The input array containing numerical values to be clipped.

### Outputs
- out: The clipped array with all values constrained within the defined range.

  </details>

<details><summary>&emsp;Function</summary>

## Function
```
Inputs:
  - array: ARRAY

Outputs:
  - out: ARRAY
```

This node applies a specified element-wise mathematical function to the input array data, processing each element independently.

### Inputs
- array: The input array of data to be processed.

### Outputs
- out: The output array after applying the selected mathematical function to each element of the input array.

  </details>

<details><summary>&emsp;Join</summary>

## Join
```
Inputs:
  - a: ARRAY
  - b: ARRAY

Outputs:
  - out: ARRAY
```

This node combines two array inputs into a single array output. It supports two methods of combining: joining the arrays along an existing axis or stacking them along a new dimension. The node manages the merging or updating of metadata from both inputs as needed.

### Inputs
- a: The first input array and its associated metadata.
- b: The second input array and its associated metadata.

### Outputs
- out: The combined array resulting from joining or stacking the two input arrays, along with updated metadata.

  </details>

<details><summary>&emsp;Math</summary>

## Math
```
Inputs:
  - data: ARRAY

Outputs:
  - out: ARRAY
```

This node performs a series of mathematical operations and rescaling on array data. It processes incoming arrays by applying configurable arithmetic operations, rounding, and optional power functions, and then maps the result from a specified input range to an output range. This allows flexible numerical manipulation and transformation of real-time signals.

### Inputs
- data: An array containing the signal or values to be processed.

### Outputs
- out: The processed array after all mathematical operations and rescaling, along with its metadata.

  </details>

<details><summary>&emsp;Operation</summary>

## Operation
```
Inputs:
  - a: ARRAY
  - b: ARRAY

Outputs:
  - out: ARRAY
```

This node performs a specified element-wise or matrix operation on two input arrays. It supports common arithmetic operations such as addition, subtraction, multiplication, division, matrix multiplication, maximum, minimum, average, and cosine similarity. The operation is applied to the data from the two input arrays, and the output retains the metadata from the first input, adjusting dimensions and sampling frequency as needed.

### Inputs
- a: The first input array, with associated metadata including dimensions and channels.
- b: The second input array, with associated metadata.

### Outputs
- out: The result of applying the selected operation to the two input arrays, along with the merged and adjusted metadata.

  </details>

<details><summary>&emsp;Reduce</summary>

## Reduce
```
Inputs:
  - array: ARRAY

Outputs:
  - out: ARRAY
```

Reduces an input array along a specified axis using a selected reduction operation. The node supports various reduction methods such as mean, median, min, max, standard deviation, norm, and sum. After reduction, it also updates the metadata to reflect changes in dimensionality.

### Inputs
- array: Multidimensional array data to be reduced.

### Outputs
- out: The array after reduction and its updated metadata.

  </details>

<details><summary>&emsp;Reshape</summary>

## Reshape
```
Inputs:
  - array: ARRAY

Outputs:
  - out: ARRAY
```

Reshapes an input array to a specified shape. The output array will have the new shape while preserving the original data. Metadata channels are removed in the output.

### Inputs
- array: The array to be reshaped.

### Outputs
- out: The reshaped array with updated metadata.

  </details>

<details><summary>&emsp;Select</summary>

## Select
```
Inputs:
  - data: ARRAY

Outputs:
  - out: ARRAY
```

Selects a subset of channels or indices along a specified axis of the input array. This node can use channel names from metadata (if present) or fall back to selecting by numerical indices. The selection can be based on inclusion or exclusion lists, and supports both named and indexed axes. The result is a reduced array with only the selected entries, and the corresponding metadata is updated accordingly.

### Inputs
- data: An array containing the data to be subselected, with optional channel metadata in the form of named axes.

### Outputs
- out: The resulting array containing only the selected channels or indices, with the metadata updated to reflect the new selection.

  </details>

<details><summary>&emsp;Transpose</summary>

## Transpose
```
Inputs:
  - array: ARRAY

Outputs:
  - out: ARRAY
```

Transposes a 2D array input, swapping its rows and columns. If the input is a 1D array, it is first converted to a 2D column vector before transposing. The node also swaps the associated "dim0" and "dim1" channel metadata to reflect the transpose operation.

### Inputs
- array: An array input, expected to be 1D or 2D. If 1D, it is reshaped to 2D before transposing.

### Outputs
- out: The transposed array along with updated channel metadata reflecting the swapped dimensions.

  </details>

</details>

## Inputs

Nodes that provide data to the pipeline.

<details><summary>View Nodes</summary>

<details><summary>&emsp;Audiocraft</summary>

## Audiocraft
```
Inputs:
  - prompt: STRING

Outputs:
  - wav: ARRAY
```

This node generates audio waveforms from a given text prompt using pretrained generative models. It leverages AudioGen or MusicGen models to synthesize audio based on the input description.

### Inputs
- prompt: A string containing the textual description or prompt for the desired audio.

### Outputs
- wav: An array representing the generated waveform audio data for the provided prompt. The output also includes metadata for the audio's sampling rate.

  </details>

<details><summary>&emsp;AudioStream</summary>

## AudioStream
```
Inputs:

Outputs:
  - out: ARRAY
```

This node captures live audio data from an input device (such as a microphone) and outputs the audio as a NumPy array. It supports real-time operation and can optionally convert multichannel audio to mono. The output includes audio data along with its sampling frequency.

### Outputs
- out: A NumPy array containing the recorded audio data (mono or multi-channel, depending on configuration) and a metadata dictionary containing the sampling frequency ('sfreq').

  </details>

<details><summary>&emsp;ConstantArray</summary>

## ConstantArray
```
Inputs:

Outputs:
  - out: ARRAY
```

This node generates an array output whose contents and structure depend on the selected mode. It can produce a constant-valued array of any specified shape, a ring graph adjacency matrix, or a random matrix. The node can be used to create standard data arrays or generate simple graph structures for further processing in the pipeline.

### Outputs
- out: An array (NumPy ndarray). The array can be a constant array of given shape filled with a specified value, a ring graph adjacency matrix, or a random matrix, depending on configuration. The output includes metadata with the sample frequency if available.

  </details>

<details><summary>&emsp;ConstantString</summary>

## ConstantString
```
Inputs:

Outputs:
  - out: STRING
```

This node outputs a constant string value whenever it is triggered. The output string is defined by the node's configuration and does not change unless the configuration is updated.

### Outputs
- out: The constant string value generated by the node.

  </details>

<details><summary>&emsp;ConstantTable</summary>

## ConstantTable
```
Inputs:

Outputs:
  - table: TABLE
```

This node generates a constant table containing up to five key-value pairs, where each key is a user-defined name and each value is either a string or a numerical array, as specified by the user. The node does not require any inputs and always outputs the configured table whenever its parameters are updated.

### Outputs
- table: A table (dictionary) where each key corresponds to a user-specified name and each value is either an array or a string, depending on the chosen data type for each entry.

  </details>

<details><summary>&emsp;EEGRecording</summary>

## EEGRecording
```
Inputs:

Outputs:
```

Streams EEG recordings as an LSL (Lab Streaming Layer) stream, either from an example dataset, a user-provided file, or a supported MNE-compatible format. This node manages reading, looping, and live replay of EEG data but does not process or modify incoming data.

### Inputs

### Outputs

  </details>

<details><summary>&emsp;ExtendedTable</summary>

## ExtendedTable
```
Inputs:
  - base: TABLE
  - array_input1: ARRAY
  - array_input2: ARRAY
  - array_input3: ARRAY
  - array_input4: ARRAY
  - array_input5: ARRAY
  - string_input1: STRING
  - string_input2: STRING
  - string_input3: STRING
  - string_input4: STRING
  - string_input5: STRING

Outputs:
  - table: TABLE
```

This node extends an input table by adding up to ten new entries, five from array inputs and five from string inputs. For each input, if a value is provided, it is added to the resulting table with a corresponding key. If no new entries are provided, the original base table is returned unchanged.

### Inputs
- base: The original table to extend.
- array_input1: First array to add to the table.
- array_input2: Second array to add to the table.
- array_input3: Third array to add to the table.
- array_input4: Fourth array to add to the table.
- array_input5: Fifth array to add to the table.
- string_input1: First string to add to the table.
- string_input2: Second string to add to the table.
- string_input3: Third string to add to the table.
- string_input4: Fourth string to add to the table.
- string_input5: Fifth string to add to the table.

### Outputs
- table: The extended table containing the original entries plus any new arrays and strings added.

  </details>

<details><summary>&emsp;FractalImage</summary>

## FractalImage
```
Inputs:
  - complexity: ARRAY

Outputs:
  - image: ARRAY
```

Generates a 2D fractal noise image based on the Fractal Brownian Motion (fBm) algorithm using Perlin noise. The node procedurally creates fractal patterns which can vary in complexity and visual detail. This can be used for texture synthesis, background generation, or as input to other visual processing nodes.

### Inputs
- complexity: If provided, overrides the default noise persistence value to modulate the fractal detail. Expects a scalar or array controlling the fBm persistence.

### Outputs
- image: A 2D array representing the generated fractal noise image, with values normalized to the [0, 1] range.

  </details>

<details><summary>&emsp;ImageGeneration</summary>

## ImageGeneration
```
Inputs:
  - prompt: STRING
  - negative_prompt: STRING
  - base_image: ARRAY

Outputs:
  - img: ARRAY
```

This node performs AI-based image generation by interfacing with either local Stable Diffusion (SD) or cloud-based DALL-E models. It supports both text-to-image (txt2img) and image-to-image (img2img) generation depending on model capabilities and user settings. The node can automatically manage image state for iterative workflows, and optionally saves generated images to disk.

### Inputs
- prompt: The main text prompt describing the desired content of the generated image.
- negative_prompt: An optional text prompt specifying aspects to avoid in the generated image.
- base_image: An optional input image used as a starting point or reference for image-to-image generation.

### Outputs
- img: The generated image as a NumPy array, along with associated prompt information.

  </details>

<details><summary>&emsp;Kuramoto</summary>

## Kuramoto
```
Inputs:
  - initial_phases: ARRAY

Outputs:
  - phases: ARRAY
  - coupling: ARRAY
  - order_parameter: ARRAY
  - waveforms: ARRAY
```

This node simulates the Kuramoto model of coupled phase oscillators, a mathematical model used to study synchronization phenomena. It integrates the phases of multiple oscillators over time, accounting for their natural frequencies and coupling, and outputs various aspects of the evolving system including the phase trajectories, the instantaneous coupling term, the global order parameter, and the corresponding oscillator waveforms.

### Inputs
- initial_phases: An array containing the initial phase values for each oscillator. If not provided, random initial phases are used.

### Outputs
- phases: The final phase values of each oscillator after simulation.
- coupling: The last computed coupling term for each oscillator.
- order_parameter: The global synchronization measure (order parameter) at the final timestep.
- waveforms: The time series waveforms for each oscillator based on their phase evolution.

  </details>

<details><summary>&emsp;LoadFile</summary>

## LoadFile
```
Inputs:
  - file: STRING

Outputs:
  - data_output: ARRAY
  - string_output: STRING
```

This node loads data from a file and outputs the loaded data in array or string format, depending on the file type and content. It supports various file types such as spectrum data, time series, generic numpy arrays, embedding CSV files, and audio files. The node processes the input file according to the specified type and outputs the corresponding data structure with optional metadata.

### Inputs
- file: The filename (as a string) of the file to load.

### Outputs
- data_output: The primary data loaded from the file, such as an array with optional metadata depending on the file type.
- string_output: The string representation of the data if applicable (e.g., if data is non-numeric and cannot be converted to an array).

  </details>

<details><summary>&emsp;LSLClient</summary>

## LSLClient
```
Inputs:
  - source_name: STRING
  - stream_name: STRING

Outputs:
  - out: ARRAY
```

This node connects to a Lab Streaming Layer (LSL) stream and receives real-time data from it. It discovers available LSL streams on the network, connects to the specified source and stream, reads chunks of incoming data, and outputs this data along with relevant metadata such as channel names and sampling frequency. The node is suitable for live signal acquisition from any source that publishes data via LSL.

### Inputs
- source_name: The LSL source ID to connect to.
- stream_name: The LSL stream name within the specified source.

### Outputs
- out: The acquired data as an array, along with metadata including sampling frequency and channel names.

  </details>

<details><summary>&emsp;MeteoMedia</summary>

## MeteoMedia
```
Inputs:
  - latitude: ARRAY
  - longitude: ARRAY
  - location_name: STRING

Outputs:
  - weather_data_table: TABLE
```

This node fetches current weather data for a given geographic location using the Tomorrow.io Realtime Weather API. It accepts either latitude and longitude coordinates or a location name as input, and outputs a table containing various weather parameters for the specified location.

### Inputs
- latitude: A single-value array representing the latitude coordinate of the location.
- longitude: A single-value array representing the longitude coordinate of the location.
- location_name: A string representing the human-readable name of the location (optional; used instead of coordinates if provided).

### Outputs
- weather_data_table: A table (dictionary of arrays) containing weather parameters such as temperature, humidity, wind speed, and other data returned by the API. If the API call fails, an "ERROR" field with the HTTP status code is provided.

  </details>

<details><summary>&emsp;Oscillator</summary>

## Oscillator
```
Inputs:
  - frequency: ARRAY

Outputs:
  - out: ARRAY
```

This node generates basic oscillator waveforms such as sine, square, sawtooth, and pulse at a specified frequency and sampling rate. It produces a continuous stream of samples corresponding to the selected waveform. The frequency of the oscillator can be controlled in real-time by providing an input array; otherwise, a set frequency parameter is used.

### Inputs
- frequency: Optional. An array containing the frequency (Hz) with which to generate samples. If not provided, a default frequency is used.

### Outputs
- out: An array of generated waveform samples, along with metadata that includes the sampling frequency.

  </details>

<details><summary>&emsp;OSCIn</summary>

## OSCIn
```
Inputs:

Outputs:
  - message: TABLE
```

Receives incoming OSC (Open Sound Control) messages over the network and makes them available as output. Each OSC message received is stored and organized by its address. Messages containing string data are output as strings, while other types are represented as arrays. This node acts as a bridge between OSC sources (such as sensors, controllers, or other software) and Goofi-Pipe, enabling real-time signal and data integration.

### Outputs
- message: A table containing the latest received OSC messages, organized by address. Each entry holds the received data, which may be a string or an array, depending on the message content.

  </details>

<details><summary>&emsp;PromptBook</summary>

## PromptBook
```
Inputs:
  - input_prompt: STRING

Outputs:
  - out: STRING
```

This node provides a standardized set of highly specialized prompts for creative text and text-to-image tasks. It allows you to select from a large collection of prompt templates designed for poetry, artistic or scientific descriptions, narratives, symbolism, horoscopes, and various text-to-image generation scenarios. The node outputs the selected prompt, optionally appending the input text to further specify or inform the prompt for downstream nodes or systems.

### Inputs
- input_prompt: Optional string to append to the selected prompt template, used to specialize or guide the generated output.

### Outputs
- out: The constructed prompt string based on the selected template and any provided input text.

  </details>

<details><summary>&emsp;RandomArray</summary>

## RandomArray
```
Inputs:

Outputs:
  - random_array: ARRAY
```

This node generates a random array of specified dimensions using either a uniform or normal distribution. Optionally, if the array is square, it can normalize the largest eigenvalue to 1. The generated array can be reset or regenerated as needed.

### Outputs
- random_array: The generated random array based on the selected distribution and dimensions.

  </details>

<details><summary>&emsp;Replay</summary>

## Replay
```
Inputs:

Outputs:
  - table_output: TABLE
```

This node replays data from a CSV file as a table, outputting one row at a time on each process step. It reads the specified CSV file, converts each row into a dictionary with appropriate data formats (including lists as NumPy arrays), and sequentially outputs the data row by row, looping back to the start after reaching the end of the file.

### Outputs
- table_output: A tuple containing a dictionary representation of the current CSV row and an empty dictionary. All columns in the CSV are included as fields in the output dictionary, with lists automatically converted to NumPy arrays where applicable.

  </details>

<details><summary>&emsp;Reservoir</summary>

## Reservoir
```
Inputs:
  - connectivity: ARRAY

Outputs:
  - data: ARRAY
```

This node implements a recurrent neural reservoir module. It maintains an internal state vector that evolves over time according to a configurable nonlinear transformation and a connectivity (weight) matrix. At each step, the state is updated by multiplying it with the connectivity matrix, optionally adding a bias, and applying a nonlinear activation function. This dynamic transformation is typically useful for tasks involving temporal or sequential data processing.

### Inputs
- connectivity: An optional square connectivity matrix (as an array) matching the size of the reservoir. If provided, it determines how each node in the reservoir is connected to others.

### Outputs
- data: The current state vector of the reservoir, after updating and applying the activation function. The output includes the array and associated sample frequency information.

  </details>

<details><summary>&emsp;SerialStream</summary>

## SerialStream
```
Inputs:

Outputs:
  - out: ARRAY
```

This node streams data directly from a serial device, supporting both ECG and capacitive sensing protocols. It reads incoming data packets, decodes and resamples them in real time to provide a uniformly sampled output array suitable for further processing.

### Outputs
- out: A NumPy array containing the decoded and resampled data from the serial device. For ECG, this is a 1D array of signal samples. For capacitive protocol, this is a 2D array where each row corresponds to a channel. The output includes metadata specifying the sampling frequency.

  </details>

<details><summary>&emsp;SimulatedEEG</summary>

## SimulatedEEG
```
Inputs:
  - exponents: ARRAY
  - peaks: ARRAY
  - variances: ARRAY
  - peak_amplitudes: ARRAY

Outputs:
  - eeg_signal: ARRAY
```

This node simulates multi-channel EEG signals in real time, generating synthetic data with controllable background spectral exponents, oscillatory peaks, variances, and peak amplitudes per channel. The generated signal is output in chunks, suitable for real-time or streaming processing.

### Inputs
- exponents: Array of spectral exponent values, one per channel, controlling the power-law background of each channel's signal.
- peaks: Array specifying oscillatory peak frequencies for each channel.
- variances: Array specifying the variance of the background component for each channel.
- peak_amplitudes: Array specifying the amplitude for each peak frequency in each channel.

### Outputs
- eeg_signal: Multichannel EEG signal data chunked over time, along with metadata specifying the sampling frequency.

  </details>

<details><summary>&emsp;SpikingNetwork</summary>

## SpikingNetwork
```
Inputs:
  - input: ARRAY

Outputs:
  - potentials: ARRAY
```

This node simulates a spiking neural network (SNN) with user-configurable network topology and real-time updating of neuron dynamics. Each time it receives an input, it integrates the input into the potentials of the neural population, executes one simulation step, and outputs the updated neuron membrane potentials. The SNN simulates spiking activity, action potential propagation, synaptic transmission delays, and refractory dynamics in a spatially organized network.

### Inputs
- input: A one-dimensional array of values, representing external input currents or signals delivered to the neuron population.

### Outputs
- potentials: The current membrane potentials of all simulated neurons in a one-dimensional array, with associated sample frequency information.

  </details>

<details><summary>&emsp;Table</summary>

## Table
```
Inputs:
  - base: TABLE
  - new_entry: ARRAY

Outputs:
  - table: TABLE
```

Creates or updates a table by adding a new entry under a specified key. If no base table is provided, an empty table is used. If no new entry is given, the current table is returned unchanged.

### Inputs
- base: The existing table to which a new entry can be added.
- new_entry: The data (as an array) to insert into the table.

### Outputs
- table: The updated table after adding the new entry, or the original table if no new entry was given.

  </details>

<details><summary>&emsp;TextGeneration</summary>

## TextGeneration
```
Inputs:
  - prompt: STRING

Outputs:
  - generated_text: STRING
```

This node performs text generation using various large language models (LLMs) from different providers, including OpenAI GPT, Anthropic Claude, Google Gemini, Ollama, and optionally a local model endpoint. It sends an input prompt to the selected LLM and returns the generated text response. Conversation history and system prompts can be used depending on the model type, and generated conversations can be saved to a file if enabled.

### Inputs
- prompt: The input text prompt, typically a user message or instruction to be completed or responded to by the language model.

### Outputs
- generated_text: The generated response text produced by the selected language model, based on the input prompt and optional conversation history.

  </details>

<details><summary>&emsp;VectorDB</summary>

## VectorDB
```
Inputs:
  - input_vector: ARRAY

Outputs:
  - top_labels: TABLE
  - vectors: ARRAY
```

This node performs vector similarity search using a pre-built vector database. Given an input vector, it retrieves the top matching vectors from the database along with their associated labels and distances. This allows you to find which entries in the database are most similar to a given input embedding.

### Inputs
- input_vector: The vector to search for in the database. Should be a 1D array representing an embedding.

### Outputs
- top_labels: A table containing the top matching labels from the database along with their distances to the input vector.
- vectors: An array of the vectors corresponding to the top matches in the database.

  </details>

<details><summary>&emsp;VideoStream</summary>

## VideoStream
```
Inputs:

Outputs:
  - frame: ARRAY
```

This node captures video frames from either a connected camera device or the desktop screen. It outputs the captured frame as a normalized RGB array, with optional mirroring and cropping applied.

### Outputs
- frame: The captured image frame as a NumPy array of shape (height, width, 3) in RGB format, with pixel values in the range [0, 1].

  </details>

<details><summary>&emsp;ZeroMQIn</summary>

## ZeroMQIn
```
Inputs:

Outputs:
  - data: ARRAY
```

This node receives data from a ZeroMQ socket using the PAIR pattern and provides this data as output in real time. It connects to a specified address and port, then waits for incoming Python objects sent via ZeroMQ, which it passes on for processing by other nodes.

### Outputs
- data: The data object received from the ZeroMQ socket, provided as an array.

  </details>

</details>

## Misc

Miscellaneous nodes that do not fit into other categories.

<details><summary>View Nodes</summary>

<details><summary>&emsp;AppendTables</summary>

## AppendTables
```
Inputs:
  - table1: TABLE
  - table2: TABLE

Outputs:
  - output_table: TABLE
```

This node combines two tables by merging their data and metadata dictionaries into a single output table. If only one input table is provided, it passes that table to the output unchanged. If both inputs are absent, the output is None.

### Inputs
- table1: The first input table to be combined.
- table2: The second input table to be combined.

### Outputs
- output_table: The resulting table containing merged data and metadata from both input tables. If only one table is provided, it outputs that table as is.

  </details>

<details><summary>&emsp;ArrayAwait</summary>

## ArrayAwait
```
Inputs:
  - data: ARRAY
  - trigger: ARRAY

Outputs:
  - out: ARRAY
```

Waits for an incoming trigger to pass through the most recent array data. The node outputs the array only when both new data is present and a trigger event occurs. If the trigger input is not received, the array data will not be forwarded to the output.

### Inputs
- data: Array to be held until a trigger is received.
- trigger: Array input used solely as a trigger event to release the latest data.

### Outputs
- out: The latest array data passed through when a trigger is received.

  </details>

<details><summary>&emsp;ColorEnhancer</summary>

## ColorEnhancer
```
Inputs:
  - image: ARRAY

Outputs:
  - enhanced_image: ARRAY
```

Enhances a color or grayscale image by applying a combination of contrast, brightness, gamma correction, and optional color boosting. The node expects the input image data to be normalized to the [0, 1] range for correct operation.

### Inputs
- image: A NumPy array representing the input image, either grayscale or color (with 3 channels), with values normalized between 0 and 1.

### Outputs
- enhanced_image: The processed image array after enhancement, with the same shape as the input, and values clipped to the [0, 1] range.

  </details>

<details><summary>&emsp;EdgeDetector</summary>

## EdgeDetector
```
Inputs:
  - image: ARRAY

Outputs:
  - edges: ARRAY
```

This node performs edge detection on input images using either the Canny or Sobel method. It receives an image array, converts it to grayscale if necessary, applies the selected edge detection algorithm, and outputs the resulting edge map as a normalized array.

### Inputs
- image: Input image array, which can be a grayscale or color image.

### Outputs
- edges: Array representing the detected edges, normalized to the range [0, 1].

  </details>

<details><summary>&emsp;FormatString</summary>

## FormatString
```
Inputs:
  - input_string_1: STRING
  - input_string_2: STRING
  - input_string_3: STRING
  - input_string_4: STRING
  - input_string_5: STRING
  - input_string_6: STRING
  - input_string_7: STRING
  - input_string_8: STRING
  - input_string_9: STRING
  - input_string_10: STRING

Outputs:
  - output_string: STRING
```

This node combines multiple input strings into a single output string, optionally using a formatting pattern. If no pattern is provided, all input strings are joined together with spaces. If a pattern is given, placeholders within the pattern are replaced with the corresponding input string values. Unnamed placeholders are filled with unused input strings in the order they appear.

### Inputs
- input_string_1: Input string to be included in the output.
- input_string_2: Input string to be included in the output.
- input_string_3: Input string to be included in the output.
- input_string_4: Input string to be included in the output.
- input_string_5: Input string to be included in the output.
- input_string_6: Input string to be included in the output.
- input_string_7: Input string to be included in the output.
- input_string_8: Input string to be included in the output.
- input_string_9: Input string to be included in the output.
- input_string_10: Input string to be included in the output.

### Outputs
- output_string: The resulting formatted string.

  </details>

<details><summary>&emsp;HSVtoRGB</summary>

## HSVtoRGB
```
Inputs:
  - hsv_image: ARRAY

Outputs:
  - rgb_image: ARRAY
```

This node converts images from the HSV (Hue, Saturation, Value) color space to the RGB (Red, Green, Blue) color space. Each pixel in the input image is transformed so that its HSV values are mapped to the corresponding RGB values.

### Inputs
- hsv_image: A NumPy array representing an image in HSV color space, where the last dimension contains the H, S, and V channels.

### Outputs
- rgb_image: A NumPy array representing the input image converted to RGB color space, with the last dimension containing the R, G, and B channels.

  </details>

<details><summary>&emsp;JoinString</summary>

## JoinString
```
Inputs:
  - string1: STRING
  - string2: STRING
  - string3: STRING
  - string4: STRING
  - string5: STRING

Outputs:
  - output: STRING
```

Joins up to five input strings into a single string, using a configurable separator. Only selected and non-empty inputs are included in the concatenation.

### Inputs
- string1: The first string to join.
- string2: The second string to join.
- string3: The third string to join.
- string4: The fourth string to join.
- string5: The fifth string to join.

### Outputs
- output: The concatenated string result.

  </details>

<details><summary>&emsp;LatentRotator</summary>

## LatentRotator
```
Inputs:
  - latent_vector: ARRAY
  - angles: ARRAY

Outputs:
  - rotated_vector: ARRAY
```

This node incrementally rotates a latent vector in its high-dimensional space based on an array of input angles. For each process step, a delta vector is calculated using the cosine of each angle and is added to the latent vector, enabling gradual, controlled movement within the latent space. The internal state accumulates these changes over time.

### Inputs
- latent_vector: The initial or current latent vector to be rotated, provided as a 1D array.
- angles: An array of angles (one per latent vector dimension) that determines the direction and magnitude of movement in the latent space.

### Outputs
- rotated_vector: The resulting latent vector after applying the incremental rotation and accumulation, as a 1D array.

  </details>

<details><summary>&emsp;NullArray</summary>

## NullArray
```
Inputs:
  - array_in: ARRAY

Outputs:
  - array_out: ARRAY
```

This node passes the input array directly to the output without modification. It is useful for situations where an explicit bypass or null operation on array data is needed.

### Inputs
- array_in: The input array data to be passed through unchanged.

### Outputs
- array_out: The unmodified array data from the input.

  </details>

<details><summary>&emsp;NullString</summary>

## NullString
```
Inputs:
  - string_in: STRING

Outputs:
  - string_out: STRING
```

This node receives a string input and passes it through unchanged to the output. It does not alter the string data or its associated metadata.

### Inputs
- string_in: The input string data to be passed through.

### Outputs
- string_out: The output string data, identical to the input.

  </details>

<details><summary>&emsp;NullTable</summary>

## NullTable
```
Inputs:
  - table_in: TABLE

Outputs:
  - table_out: TABLE
```

This node receives a table and passes it through unchanged. It does not modify the table's data or metadata. This node can be used when a direct passthrough of table data is needed in a processing graph.

### Inputs
- table_in: The input table to be passed through.

### Outputs
- table_out: The same table as the input, unchanged.

  </details>

<details><summary>&emsp;RGBtoHSV</summary>

## RGBtoHSV
```
Inputs:
  - rgb_image: ARRAY

Outputs:
  - hsv_image: ARRAY
```

This node converts an input RGB image to the HSV color space. It processes each pixel of the input image, transforming its Red, Green, Blue (RGB) values to the corresponding Hue, Saturation, Value (HSV) representation, and outputs the result as an image with the same shape as the input.

### Inputs
- rgb_image: An image array with three channels representing the Red, Green, and Blue color values for each pixel.

### Outputs
- hsv_image: An image array where each pixel contains the corresponding Hue, Saturation, and Value components derived from the input RGB values.

  </details>

<details><summary>&emsp;SetMeta</summary>

## SetMeta
```
Inputs:
  - array: ARRAY

Outputs:
  - out: ARRAY
```

This node sets a metadata key-value pair on an input array. The value is type-cast based on a specified type before assignment. The data remains unchanged.

### Inputs
- array: Input array data with accompanying metadata.

### Outputs
- out: The same array data with the updated metadata.

  </details>

<details><summary>&emsp;StringAwait</summary>

## StringAwait
```
Inputs:
  - message: STRING
  - trigger: ARRAY

Outputs:
  - out: STRING
```

Waits for a trigger signal before outputting the provided string message. The node only outputs the message in response to a trigger, and can be set to only emit when the message content has changed. Once triggered, the output is generated and the trigger is consumed.

### Inputs
- message: The string to be output when triggered.
- trigger: An array acting as the trigger signal. The presence of a value triggers the output.

### Outputs
- out: The provided string message, passed through when the trigger is received.

  </details>

<details><summary>&emsp;StringToTable</summary>

## StringToTable
```
Inputs:
  - text: STRING

Outputs:
  - table: TABLE
```

Converts a text containing structured data (in JSON or YAML format) into a Goofi table. The node parses the input string, automatically handling nested objects and converting them into nested table structures. Strings become string fields, arrays become array fields, and objects become subtables.

### Inputs
- text: The input string containing data in a structured format (JSON or YAML).

### Outputs
- table: The resulting table parsed from the input text, with fields mapped to Goofi TABLE, STRING, or ARRAY types as appropriate.

  </details>

<details><summary>&emsp;Switch</summary>

## Switch
```
Inputs:
  - selector: ARRAY
  - array1: ARRAY
  - array2: ARRAY
  - array3: ARRAY
  - string1: STRING
  - string2: STRING
  - string3: STRING

Outputs:
  - array_out: ARRAY
  - string_out: STRING
```

Selects and forwards one of several input arrays or strings to the output, based on the selector input and mode. The node routes either an array or a string input to the corresponding output.

### Inputs
- selector: Array input used to choose which input will be forwarded. The first element indicates selection (1, 2, or 3).
- array1: First array input.
- array2: Second array input.
- array3: Third array input.
- string1: First string input.
- string2: Second string input.
- string3: Third string input.

### Outputs
- array_out: Forwards the selected array input when the node is set to array mode. Only one array input is passed to the output depending on the selector value.
- string_out: Forwards the selected string input when the node is set to string mode. Only one string input is passed to the output depending on the selector value.

  </details>

<details><summary>&emsp;TableSelectArray</summary>

## TableSelectArray
```
Inputs:
  - input_table: TABLE

Outputs:
  - output_array: ARRAY
```

Selects a specified array column from an input table and outputs it as a separate array, preserving the table's metadata.

### Inputs
- input_table: Table containing one or more columns with data keyed by string.

### Outputs
- output_array: Array extracted from the input table, corresponding to the selected column, with associated metadata.

  </details>

<details><summary>&emsp;TableSelectString</summary>

## TableSelectString
```
Inputs:
  - input_table: TABLE

Outputs:
  - output_string: STRING
```

This node extracts a specific string value from an input table based on a selected key. If the value associated with the key is not already a string, it is converted to a string before output.

### Inputs
- input_table: A table containing key-value pairs where each value is a data object.

### Outputs
- output_string: The string value retrieved from the table using the selected key.

  </details>

<details><summary>&emsp;TableToString</summary>

## TableToString
```
Inputs:
  - table: TABLE

Outputs:
  - text: STRING
```

Converts a table data structure into a text representation in either JSON or YAML format. The node serializes the input table and outputs it as a string, enabling easy inspection or further text-based processing.

### Inputs
- table: The input table to be converted to text.

### Outputs
- text: The string representation of the input table, serialized in the chosen format.

  </details>

</details>

## Outputs

Nodes that send data to external systems.

<details><summary>View Nodes</summary>

<details><summary>&emsp;AudioOut</summary>

## AudioOut
```
Inputs:
  - data: ARRAY
  - device: STRING

Outputs:
  - finished: ARRAY
```

This node plays incoming audio data to an audio output device in real-time. It receives audio arrays and routes the signal to the selected audio hardware, handling transitions between consecutive audio blocks for seamless playback. The node can also switch audio devices dynamically.

### Inputs
- data: Array of audio samples to play through the output device.
- device: Name of the audio output device to use.

### Outputs
- finished: Signals completion of playback for the current audio data block.

  </details>

<details><summary>&emsp;LSLOut</summary>

## LSLOut
```
Inputs:
  - data: ARRAY

Outputs:
```

This node outputs incoming array data as a Lab Streaming Layer (LSL) stream, allowing real-time transmission of signals (such as EEG, sensor data, etc.) to other software or machines compatible with LSL. The node automatically creates and manages an LSL outlet, configuring its channels and parameters to match the input array.

### Inputs
- data: A 1D or 2D array of floating-point data to be sent over LSL. The array can represent multi-channel or single-channel time series data. The expected channel names and sample frequency may be specified in metadata.

### Outputs
- None. (This node transmits data to an external LSL stream and does not produce an output within the goofi-pipe node graph.)

  </details>

<details><summary>&emsp;MidiCCout</summary>

## MidiCCout
```
Inputs:
  - cc1: ARRAY
  - cc2: ARRAY
  - cc3: ARRAY
  - cc4: ARRAY
  - cc5: ARRAY

Outputs:
  - midi_status: STRING
```

This node converts up to five arrays of input values into MIDI Control Change (CC) messages and sends them out to a selected MIDI output port. Each input array corresponds to a configurable MIDI CC number, allowing for real-time control of MIDI devices or software via multiple CC messages. The node provides status output indicating the success or any errors during message transmission.

### Inputs
- cc1: Array of values to send as MIDI CC messages using the first configured CC number.
- cc2: Array of values to send as MIDI CC messages using the second configured CC number.
- cc3: Array of values to send as MIDI CC messages using the third configured CC number.
- cc4: Array of values to send as MIDI CC messages using the fourth configured CC number.
- cc5: Array of values to send as MIDI CC messages using the fifth configured CC number.

### Outputs
- midi_status: Status message indicating whether the MIDI CC messages were sent successfully or if errors occurred.

  </details>

<details><summary>&emsp;MidiOut</summary>

## MidiOut
```
Inputs:
  - note: ARRAY
  - velocity: ARRAY
  - duration: ARRAY

Outputs:
  - midi_status: STRING
```

This node sends MIDI note messages to an external MIDI device or software instrument. It receives note numbers (or frequencies), velocities, and durations as input arrays, and sends the corresponding MIDI note on/off messages using the selected output port and channel. Notes can be played either simultaneously or sequentially. The status of the MIDI message transmission is output as a string.

### Inputs
- note: An array of MIDI note numbers to play, or frequencies if Hz input is enabled.
- velocity: An array of velocities for each note. If not provided, default velocity is used.
- duration: An array of durations in seconds for each note. If not provided, default duration is used.

### Outputs
- midi_status: A string indicating whether the MIDI notes were sent successfully, or if there was an error with the note or velocity range.

  </details>

<details><summary>&emsp;OSCOut</summary>

## OSCOut
```
Inputs:
  - data: TABLE

Outputs:
```

This node sends incoming table data as OSC (Open Sound Control) messages to a specified network address and port. Input data is serialized to OSC message bundles or individual messages and transmitted, with options for broadcasting and message change detection.

### Inputs
- data: Table data to be transmitted as OSC messages.

### Outputs
- None.

  </details>

<details><summary>&emsp;SharedMemOut</summary>

## SharedMemOut
```
Inputs:
  - data: ARRAY

Outputs:
```

This node writes array data to a shared memory segment to enable efficient inter-process communication with other processes that can access the same memory. When new array data is received, it is converted to 32-bit floating point format and copied into the shared memory, overwriting any previous contents. This allows other applications or processes to read the current array data in real-time.

### Inputs
- data: An array containing floating point values to be written to shared memory.

### Outputs
- None

  </details>

<details><summary>&emsp;WriteCsv</summary>

## WriteCsv
```
Inputs:
  - table_input: TABLE
  - start: ARRAY
  - stop: ARRAY
  - fname: STRING

Outputs:
```

This node writes incoming table data to a CSV file, supporting both generic tables and EEG-specific formats. The node can append new rows to an existing CSV, generate unique filenames based on the current time, and optionally include timestamps. Two writing modes are supported: a default mode for general tabular data and an EEG mode that handles multidimensional arrays and sampling frequency metadata. Data is automatically flattened and serialized as needed to preserve structure in the CSV output.

### Inputs
- table_input: Table data to be written into the CSV file. The table can contain nested tables, arrays, or strings.
- start: Array signal triggering the start of writing to the CSV file.
- stop: Array signal triggering the stop of writing to the CSV file.
- fname: String specifying the filename to use for the CSV output.

### Outputs
- None. This node writes data to disk but does not produce downstream data outputs.

  </details>

<details><summary>&emsp;WriteCsvSafe</summary>

## WriteCsvSafe
```
Inputs:
  - data: ARRAY
  - annot: TABLE
  - start: ARRAY
  - stop: ARRAY
  - fname: STRING

Outputs:
  - status: STRING
```

  </details>

<details><summary>&emsp;ZeroMQOut</summary>

## ZeroMQOut
```
Inputs:
  - data: ARRAY

Outputs:
```

This node sends array data to an external application or process over a network connection using the ZeroMQ library. It transmits data in real time via a TCP socket, allowing integration with remote systems or distributed processing setups.

### Inputs
- data: The array data to be transmitted, which is sent as a NumPy float32 array.

### Outputs
- None.

  </details>

</details>

## Signal

Nodes implementing signal processing operations.

<details><summary>View Nodes</summary>

<details><summary>&emsp;Autocorrelation</summary>

## Autocorrelation
```
Inputs:
  - signal: ARRAY

Outputs:
  - autocorr: ARRAY
```

Computes the autocorrelation of input array signals along a specified axis. The autocorrelation measures the similarity of a signal with delayed versions of itself, and can reveal repeating patterns or periodicity in data.

### Inputs
- signal: Input array (may be one-dimensional or multi-dimensional) to compute the autocorrelation from.

### Outputs
- autocorr: The resulting autocorrelation array, with the same or reduced dimensionality depending on processing.

  </details>

<details><summary>&emsp;Buffer</summary>

## Buffer
```
Inputs:
  - val: ARRAY

Outputs:
  - out: ARRAY
```

Buffers incoming array data along a specified axis, maintaining a rolling window of the most recent samples or seconds. The buffer is updated in real-time as new data arrives, concatenating incoming arrays and discarding the oldest to keep the buffer size constant. Channel metadata is propagated and updated accordingly. The node supports resetting to clear the buffer. The output is the current buffer contents with updated metadata.

### Inputs
- val: Array data to be buffered, with associated metadata.

### Outputs
- out: The current contents of the buffer as an array, along with updated metadata.

  </details>

<details><summary>&emsp;BufferString</summary>

## BufferString
```
Inputs:
  - val: STRING

Outputs:
  - out: STRING
```

This node accumulates incoming string values into a rolling buffer and outputs the concatenated result. Each new string input is split according to a chosen separator, and the resulting pieces are appended to the buffer. If the buffer exceeds its maximum size, the oldest entries are removed to maintain the limit. The output is the joined contents of the current buffer as a single string.

### Inputs
- val: String data to be added to the buffer.

### Outputs
- out: The concatenated string of buffer contents after appending the latest input.

  </details>

<details><summary>&emsp;Cycle</summary>

## Cycle
```
Inputs:
  - signal: ARRAY

Outputs:
  - cycle: ARRAY
```

This node computes the average cycle shape of an oscillatory signal. It buffers incoming array data and, once enough data is collected, segments it into cycles based on a specified frequency. The node then averages a specified number of these cycles to create a representative cycle waveform of the input signal. This can be used to analyze the typical shape of periodic signals.

### Inputs
- signal: Array data representing the signal to be processed. The input must have a sampling frequency specified in its metadata.

### Outputs
- cycle: Array data containing the averaged cycle waveform computed from the input signal. The output has the same dimensionality as the input.

  </details>

<details><summary>&emsp;Delay</summary>

## Delay
```
Inputs:
  - data: ARRAY

Outputs:
  - output: ARRAY
```

Introduces a configurable time delay into the data stream, pausing the forwarding of incoming array data for a specified duration before outputting it unchanged.

### Inputs
- data: An array of data to be delayed.

### Outputs
- output: The same array data as received on input, passed after the specified delay.

  </details>

<details><summary>&emsp;EEGHeadsetDetection</summary>

## EEGHeadsetDetection
```
Inputs:
  - eeg_data: ARRAY

Outputs:
  - headset_status: ARRAY
  - centered_abs_data: ARRAY
```

This node detects whether an EEG headset is worn or not by analyzing incoming EEG data. It monitors the average value of the EEG signal to determine the status and also handles situations where data is missing to decide if the headset is disconnected.

### Inputs
- eeg_data: The EEG signal data provided as an array.

### Outputs
- headset_status: An array indicating the detected status of the headset: 0 if disconnected, 1 if not worn, or 2 if worn.

  </details>

<details><summary>&emsp;EMD</summary>

## EMD
```
Inputs:
  - data: ARRAY

Outputs:
  - IMFs: ARRAY
```

Applies Empirical Mode Decomposition (EMD) to a one-dimensional array input signal, extracting its intrinsic mode functions (IMFs). The node returns the resulting IMFs as a multi-channel array, with each channel corresponding to an individual IMF. IMF indices are added to the metadata for channel identification.

### Inputs
- data: A one-dimensional array representing the input signal to be decomposed.

### Outputs
- IMFs: An array containing the extracted intrinsic mode functions, with channel metadata indicating the IMF index.

  </details>

<details><summary>&emsp;FFT</summary>

## FFT
```
Inputs:
  - data: ARRAY

Outputs:
  - mag: ARRAY
  - phase: ARRAY
```

This node computes the Fast Fourier Transform (FFT) of the input data, allowing analysis of frequency components in either time series or image data. For time series, it processes 1D or 2D arrays and outputs the magnitude and phase spectra for each frequency. For images, it processes grayscale or RGB images and outputs the magnitude and phase spectra over spatial frequencies.

### Inputs
- data: The input array to be transformed, either as a time series (1D or 2D array with sampling frequency in metadata) or as an image (2D grayscale or 3D RGB array).

### Outputs
- mag: The magnitude spectrum of the FFT of the input data.
- phase: The phase spectrum of the FFT of the input data.

  </details>

<details><summary>&emsp;Filter</summary>

## Filter
```
Inputs:
  - data: ARRAY

Outputs:
  - filtered_data: ARRAY
```

This node applies digital signal filtering to incoming data arrays, allowing for the removal or attenuation of specific frequency components. The node supports real-time bandpass and notch filtering, with options for causal or zero-phase filters. It can also perform detrending and demeaning operations on the signal. Internally, it can buffer incoming data to enable more robust filtering.

### Inputs
- data: An array of numerical signal data to be filtered. The data should include associated metadata such as the sampling frequency ("sfreq").

### Outputs
- filtered_data: The filtered version of the input data array, returned with original metadata. This output contains the signal after the selected filtering and optional preprocessing steps have been applied.

  </details>

<details><summary>&emsp;FOOOFaperiodic</summary>

## FOOOFaperiodic
```
Inputs:
  - psd_data: ARRAY

Outputs:
  - offset: ARRAY
  - exponent: ARRAY
  - cf_peaks: ARRAY
  - cleaned_psd: ARRAY
```

This node extracts the aperiodic parameters and peak center frequencies from input power spectral density (PSD) data using the FOOOF algorithm. It fits FOOOF models to each spectrum, separates the aperiodic (background) component from periodic peaks, and outputs both the background parameters and the PSD with the peaks removed.

### Inputs
- psd_data: Input array containing one or more power spectra with associated frequency information.

### Outputs
- offset: The offset parameter of the aperiodic (background) component for each spectrum.
- exponent: The exponent parameter of the aperiodic (background) component for each spectrum.
- cf_peaks: The center frequencies of all detected peaks in each spectrum.
- cleaned_psd: The power spectrum with periodic peaks removed, containing the aperiodic component only.

  </details>

<details><summary>&emsp;FrequencyShift</summary>

## FrequencyShift
```
Inputs:
  - data: ARRAY

Outputs:
  - out: ARRAY
```

Shifts the frequency content of an input signal by a specified amount using the FFT frequency shifting method. The node takes a time-domain signal, converts it to the frequency domain, shifts the spectrum, and converts it back to the time domain.

### Inputs
- data: An array representing the input signal to shift. Must include metadata with the sampling frequency ("sfreq").

### Outputs
- out: The frequency-shifted signal as an array, with the original metadata preserved.

  </details>

<details><summary>&emsp;Hilbert</summary>

## Hilbert
```
Inputs:
  - data: ARRAY

Outputs:
  - inst_amplitude: ARRAY
  - inst_phase: ARRAY
  - inst_frequency: ARRAY
```

Computes the analytic signal of the input data using the Hilbert transform and extracts key instantaneous signal properties. Outputs the instantaneous amplitude, phase, and frequency of the input array, useful for advanced signal processing and time-frequency analysis.

### Inputs
- data: Input array representing one or more time-series signals.

### Outputs
- inst_amplitude: Instantaneous amplitude of the analytic signal for each input channel.
- inst_phase: Instantaneous phase of the analytic signal for each input channel.
- inst_frequency: Instantaneous frequency derived from unwrapped phase differences for each input channel.

  </details>

<details><summary>&emsp;Histogram</summary>

## Histogram
```
Inputs:
  - data: ARRAY

Outputs:
  - histogram: ARRAY
```

This node computes the histogram of an input array using either traditional binning or kernel density estimation (KDE). It receives an array of data, flattens it, computes a histogram or KDE within a specified data range, and outputs the resulting distribution along with metadata describing each bin.

### Inputs
- data: An array of numerical values to be analyzed.

### Outputs
- histogram: The computed histogram or KDE values, along with metadata including the bin lower edges as channel names.

  </details>

<details><summary>&emsp;IFFT</summary>

## IFFT
```
Inputs:
  - spectrum: ARRAY
  - phase: ARRAY

Outputs:
  - reconstructed: ARRAY
```

This node performs an inverse Fast Fourier Transform (IFFT) to reconstruct a time-domain signal from its magnitude (spectrum) and phase data. It accepts separate arrays for the spectrum and phase, combines them into complex frequency-domain data, and then computes the time-domain signal using the inverse FFT. If the input arrays differ in length, the shorter array is zero-padded to match the longer one before processing.

### Inputs
- spectrum: Array containing the magnitude (spectrum) values of the frequency-domain signal.
- phase: Array containing the phase values corresponding to the frequency-domain signal.

### Outputs
- reconstructed: Array containing the reconstructed time-domain signal obtained from the IFFT, along with the metadata from the phase input.

  </details>

<details><summary>&emsp;Normalization</summary>

## Normalization
```
Inputs:
  - data: ARRAY

Outputs:
  - normalized: ARRAY
```

This node performs normalization on array data using various normalization methods such as z-score, quantile, robust, or min-max scaling. It manages a buffer of incoming data along a specified axis to compute statistics and applies the selected normalization technique in real-time. The node is designed for use with multi-dimensional array data and is suitable for pre-processing input signals to have consistent statistical properties.

### Inputs
- data: Array input data to be normalized.

### Outputs
- normalized: The normalized array data, transformed according to the selected normalization method and matching the structure and size of the input.

  </details>

<details><summary>&emsp;PSD</summary>

## PSD
```
Inputs:
  - data: ARRAY

Outputs:
  - psd: ARRAY
```

This node computes the Power Spectral Density (PSD) of input array data using either the FFT or Welch method. It processes one- or two-dimensional input data and returns the PSD values along with frequency information, within a specified frequency range.

### Inputs
- data: An array (1D or 2D) containing the signal data to analyze, with associated metadata including sampling frequency.

### Outputs
- psd: An array representing the power spectral density of the input data, along with updated metadata including the selected frequency values.

  </details>

<details><summary>&emsp;Recurrence</summary>

## Recurrence
```
Inputs:
  - input_array: ARRAY

Outputs:
  - recurrence_matrix: ARRAY
  - RR: ARRAY
  - DET: ARRAY
  - LAM: ARRAY
```

Computes the recurrence matrix and several recurrence quantification analysis (RQA) metrics for a given input array. The recurrence matrix represents recurrent states in the data based on pairwise distances, and the RQA metrics quantify aspects of the recurrence structure, including recurrence rate, determinism, and laminarity.

### Inputs
- input_array: N-dimensional array of data points to analyze for recurrence structures.

### Outputs
- recurrence_matrix: Matrix indicating where recurrences occur in the input data.
- RR: Array containing the calculated recurrence rate.
- DET: Array containing the calculated determinism.
- LAM: Array containing the calculated laminarity.

  </details>

<details><summary>&emsp;Resample</summary>

## Resample
```
Inputs:
  - data: ARRAY

Outputs:
  - out: ARRAY
```

This node resamples an input signal array from its original sampling frequency to a new specified sampling frequency using polyphase filtering. It supports resampling along any axis of the input array. The node ensures any invalid numerical values (such as NaN or infinity) in the input are replaced with zeros before processing. The output array has its sample rate updated and, if applicable, the relevant channel metadata for the resampled axis is removed.

### Inputs
- data: Array data to be resampled, along with its associated metadata including sampling frequency and channel information.

### Outputs
- out: The resampled array with updated metadata reflecting the new sampling frequency and any changes to channel information.

  </details>

<details><summary>&emsp;ResampleJoint</summary>

## ResampleJoint
```
Inputs:
  - data1: ARRAY
  - data2: ARRAY

Outputs:
  - out1: ARRAY
  - out2: ARRAY
```

This node jointly resamples two input arrays so that both are converted to the same new sampling frequency, computed as an interpolation between their original sampling rates. The original data arrays are resampled to this common frequency and outputted along with updated metadata.

### Inputs
- data1: First input array along with its associated metadata, including original sampling frequency ("sfreq").
- data2: Second input array along with its associated metadata, including original sampling frequency ("sfreq").

### Outputs
- out1: Resampled version of data1, matched to the new common sampling frequency with updated metadata.
- out2: Resampled version of data2, matched to the new common sampling frequency with updated metadata.

  </details>

<details><summary>&emsp;Smooth</summary>

## Smooth
```
Inputs:
  - data: ARRAY

Outputs:
  - out: ARRAY
```

Applies Gaussian smoothing to an input array along a specified axis, reducing noise or fluctuations and producing a smoothed version of the original data.

### Inputs
- data: Array data to be smoothed.

### Outputs
- out: Smoothed array data, with the same shape as the input.

  </details>

<details><summary>&emsp;StaticBaseline</summary>

## StaticBaseline
```
Inputs:
  - data: ARRAY
  - trigger_baseline: ARRAY
  - n_seconds: ARRAY

Outputs:
  - normalized: ARRAY
```

This node computes a normalized version of an incoming data array using a baseline period. The baseline is accumulated over a fixed duration, after which normalization is performed using either mean-based z-scoring or quantile transformation. The baseline window is reset and accumulated anew when triggered, allowing normalization to adapt to different periods of the data stream.

### Inputs
- data: The 1D or 2D array of data to be normalized.
- trigger_baseline: Optional input. Triggers a reset of the baseline window when any value in the array is greater than zero.
- n_seconds: Optional input. Specifies the number of seconds to use for accumulating the baseline window.

### Outputs
- normalized: The normalized array using statistics from the accumulated baseline window, matching the shape of the input data.

  </details>

<details><summary>&emsp;TableNormalization</summary>

## TableNormalization
```
Inputs:
  - table: TABLE

Outputs:
  - normalized: TABLE
```

This node performs normalization on each column of a table independently. For each key (column) in the input table, it applies a normalization process and produces a table with normalized data columns. Each column is normalized using its own normalization state, ensuring independent treatment per column.

### Inputs
- table: A table containing multiple columns (keys), where each column will be normalized separately.

### Outputs
- normalized: A table with the same structure as the input, where each column has been normalized.

  </details>

<details><summary>&emsp;Threshold</summary>

## Threshold
```
Inputs:
  - data: ARRAY

Outputs:
  - thresholded: ARRAY
```

Applies a thresholding operation to an input array. The node compares each value in the input array to a specified threshold using one of several modes (e.g., greater than, less than), and outputs an array where each element is set to a true or false value depending on the comparison result. The node supports features such as minimum delay between triggers, retrigger requirements, and outputting NaN after certain threshold crossings.

### Inputs
- data: The input array to be thresholded.

### Outputs
- thresholded: The thresholded array with elements set according to whether the input values passed the threshold condition.

  </details>

<details><summary>&emsp;TimeDelayEmbedding</summary>

## TimeDelayEmbedding
```
Inputs:
  - input_array: ARRAY

Outputs:
  - embedded_array: ARRAY
```

This node performs time delay embedding on an input array. Time delay embedding reconstructs the state space of a signal by creating multiple delayed copies of the input and stacking them together, enabling the analysis of temporal structures and dynamics within the data.

### Inputs
- input_array: A one-dimensional array representing the input signal to be embedded.

### Outputs
- embedded_array: An array where each element contains values from the original signal at different delay steps, enabling state space reconstruction, along with updated metadata.

  </details>

<details><summary>&emsp;WelfordsZTransform</summary>

## WelfordsZTransform
```
Inputs:
  - data: ARRAY

Outputs:
  - normalized: ARRAY
```

This node performs online z-score normalization of input array data using Welford's algorithm for numerically stable, running computation of mean and standard deviation. It updates normalization statistics incrementally as new values arrive, allowing for continual, real-time signal normalization without buffering the entire input stream. The node handles both 1D and 2D arrays, normalizing each individual signal channel independently. Initially, output values are set to zero until sufficient samples have been collected for reliable statistics. Strong outliers are clipped to within a configurable range.

### Inputs
- data: 1D or 2D array of numerical values to be normalized in real time.

### Outputs
- normalized: Array of the same shape as input, containing the online z-score normalized values.

  </details>

</details>
<!-- !!GOOFI_PIPE_NODE_LIST_END!! -->
