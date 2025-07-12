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

<h3 align="center"><a href="https://notebooklm.google.com/notebook/b28ee248-db68-4b41-b914-1fc8312f86a8/audio?pli=1">AI Podcast about goofi-pipe</a> | <a href="https://deepwiki.com/dav0dea/goofi-pipe">AI-generated Documentation</a></h3>

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

  - **Inputs:**
    - prompt: InputSlot(dtype=<DataType.STRING: 1>, trigger_process=False, data=None)
    - audio: ARRAY
  - **Outputs:**
    - generated_text: STRING
  </details>

<details><summary>&emsp;AudioTagging</summary>

  - **Inputs:**
    - audioIn: ARRAY
  - **Outputs:**
    - tags: STRING
    - probabilities: ARRAY
    - embedding: ARRAY
  </details>

<details><summary>&emsp;Avalanches</summary>

  - **Inputs:**
    - data: ARRAY
  - **Outputs:**
    - size: ARRAY
    - duration: ARRAY
  </details>

<details><summary>&emsp;Binarize</summary>

  - **Inputs:**
    - data: ARRAY
  - **Outputs:**
    - bin_data: ARRAY
  </details>

<details><summary>&emsp;Bioelements</summary>

  - **Inputs:**
    - data: ARRAY
  - **Outputs:**
    - elements: TABLE
  </details>

<details><summary>&emsp;Bioplanets</summary>

  - **Inputs:**
    - peaks: ARRAY
  - **Outputs:**
    - planets: TABLE
    - top_planets: STRING
  </details>

<details><summary>&emsp;Biorhythms</summary>

  - **Inputs:**
    - tuning: ARRAY
  - **Outputs:**
    - pulses: ARRAY
    - steps: ARRAY
    - offsets: ARRAY
  </details>

<details><summary>&emsp;Biotuner</summary>

  - **Inputs:**
    - data: ARRAY
  - **Outputs:**
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
  </details>

<details><summary>&emsp;CardiacRespiration</summary>

  - **Inputs:**
    - data: ARRAY
  - **Outputs:**
    - cardiac: ARRAY
  </details>

<details><summary>&emsp;CardioRespiratoryVariability</summary>

  - **Inputs:**
    - data: ARRAY
  - **Outputs:**
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
  </details>

<details><summary>&emsp;Classifier</summary>

  - **Inputs:**
    - data: ARRAY
  - **Outputs:**
    - probs: ARRAY
    - feature_importances: ARRAY
  </details>

<details><summary>&emsp;Clustering</summary>

  - **Inputs:**
    - matrix: ARRAY
  - **Outputs:**
    - cluster_labels: ARRAY
    - cluster_centers: ARRAY
  </details>

<details><summary>&emsp;Compass</summary>

  - **Inputs:**
    - pole1: ARRAY
    - pole2: ARRAY
  - **Outputs:**
    - angles: ARRAY
  </details>

<details><summary>&emsp;Connectivity</summary>

  - **Inputs:**
    - data: ARRAY
  - **Outputs:**
    - matrix: ARRAY
  </details>

<details><summary>&emsp;Coord2loc</summary>

  - **Inputs:**
    - latitude: ARRAY
    - longitude: ARRAY
  - **Outputs:**
    - coord_info: TABLE
    - water_situation: ARRAY
  </details>

<details><summary>&emsp;Correlation</summary>

  - **Inputs:**
    - data1: ARRAY
    - data2: ARRAY
  - **Outputs:**
    - pearson: ARRAY
    - pval: ARRAY
  </details>

<details><summary>&emsp;DimensionalityReduction</summary>

  - **Inputs:**
    - data: ARRAY
    - new_data: ARRAY
  - **Outputs:**
    - transformed: ARRAY
    - new_components: ARRAY
  </details>

<details><summary>&emsp;DissonanceCurve</summary>

  - **Inputs:**
    - peaks: ARRAY
    - amps: ARRAY
  - **Outputs:**
    - dissonance_curve: ARRAY
    - tuning: ARRAY
    - avg_dissonance: ARRAY
  </details>

<details><summary>&emsp;DreamInceptor</summary>

  - **Inputs:**
    - data: ARRAY
    - start: ARRAY
    - reset: ARRAY
  - **Outputs:**
    - trigger: ARRAY
    - z_theta_alpha: ARRAY
    - z_lempel_ziv: ARRAY
    - baseline_stats: TABLE
  </details>

<details><summary>&emsp;EEGEmbedding</summary>

  - **Inputs:**
    - eeg: ARRAY
  - **Outputs:**
    - embeddings: ARRAY
  </details>

<details><summary>&emsp;EigenDecomposition</summary>

  - **Inputs:**
    - matrix: ARRAY
  - **Outputs:**
    - eigenvalues: ARRAY
    - eigenvectors: ARRAY
  </details>

<details><summary>&emsp;Embedding</summary>

  - **Inputs:**
    - text: STRING
    - data: ARRAY
  - **Outputs:**
    - text_embeddings: ARRAY
    - data_embeddings: ARRAY
  </details>

<details><summary>&emsp;ERP</summary>

  - **Inputs:**
    - signal: ARRAY
    - trigger: ARRAY
  - **Outputs:**
    - erp: ARRAY
  </details>

<details><summary>&emsp;FacialExpression</summary>

  - **Inputs:**
    - image: ARRAY
  - **Outputs:**
    - emotion_probabilities: ARRAY
    - action_units: ARRAY
    - main_emotion: STRING
  </details>

<details><summary>&emsp;Fractality</summary>

  - **Inputs:**
    - data_input: ARRAY
  - **Outputs:**
    - fractal_dimension: ARRAY
  </details>

<details><summary>&emsp;GraphMetrics</summary>

  - **Inputs:**
    - matrix: ARRAY
  - **Outputs:**
    - clustering_coefficient: ARRAY
    - characteristic_path_length: ARRAY
    - betweenness_centrality: ARRAY
    - degree_centrality: ARRAY
    - assortativity: ARRAY
    - transitivity: ARRAY
  </details>

<details><summary>&emsp;HarmonicSpectrum</summary>

  - **Inputs:**
    - psd: ARRAY
  - **Outputs:**
    - harmonic_spectrum: ARRAY
    - max_harmonicity: ARRAY
    - avg_harmonicity: ARRAY
  </details>

<details><summary>&emsp;Img2Txt</summary>

  - **Inputs:**
    - image: ARRAY
  - **Outputs:**
    - generated_text: STRING
  </details>

<details><summary>&emsp;LempelZiv</summary>

  - **Inputs:**
    - data: ARRAY
  - **Outputs:**
    - lzc: ARRAY
  </details>

<details><summary>&emsp;Monolith</summary>

  - **Inputs:**
    - data: ARRAY
  - **Outputs:**
    - features: ARRAY
  </details>

<details><summary>&emsp;PCA</summary>

  - **Inputs:**
    - data: ARRAY
  - **Outputs:**
    - principal_components: ARRAY
  </details>

<details><summary>&emsp;PhiID</summary>

  - **Inputs:**
    - matrix: ARRAY
  - **Outputs:**
    - PhiID: ARRAY
    - inf_dyn: ARRAY
    - IIT: ARRAY
  </details>

<details><summary>&emsp;PoseEstimation</summary>

  - **Inputs:**
    - image: ARRAY
  - **Outputs:**
    - pose: ARRAY
  </details>

<details><summary>&emsp;PowerBand</summary>

  - **Inputs:**
    - data: ARRAY
  - **Outputs:**
    - power: ARRAY
  </details>

<details><summary>&emsp;PowerBandEEG</summary>

  - **Inputs:**
    - data: ARRAY
  - **Outputs:**
    - delta: ARRAY
    - theta: ARRAY
    - alpha: ARRAY
    - lowbeta: ARRAY
    - highbeta: ARRAY
    - gamma: ARRAY
  </details>

<details><summary>&emsp;ProbabilityMatrix</summary>

  - **Inputs:**
    - input_data: ARRAY
  - **Outputs:**
    - data: ARRAY
  </details>

<details><summary>&emsp;ReveEEG</summary>

  - **Inputs:**
    - eeg: ARRAY
  - **Outputs:**
    - embedding: ARRAY
  </details>

<details><summary>&emsp;SpectroMorphology</summary>

  - **Inputs:**
    - data: ARRAY
  - **Outputs:**
    - spectro: ARRAY
  </details>

<details><summary>&emsp;SpeechSynthesis</summary>

  - **Inputs:**
    - text: STRING
    - voice: ARRAY
  - **Outputs:**
    - speech: ARRAY
    - transcript: STRING
  </details>

<details><summary>&emsp;TotoEmbedding</summary>

  - **Inputs:**
    - timeseries: ARRAY
  - **Outputs:**
    - embedding: ARRAY
  </details>

<details><summary>&emsp;TransitionalHarmony</summary>

  - **Inputs:**
    - data: ARRAY
  - **Outputs:**
    - trans_harm: ARRAY
    - melody: ARRAY
  </details>

<details><summary>&emsp;TuningColors</summary>

  - **Inputs:**
    - data: ARRAY
  - **Outputs:**
    - hue: ARRAY
    - saturation: ARRAY
    - value: ARRAY
    - color_names: STRING
  </details>

<details><summary>&emsp;TuningMatrix</summary>

  - **Inputs:**
    - tuning: ARRAY
  - **Outputs:**
    - matrix: ARRAY
    - metric_per_step: ARRAY
    - metric: ARRAY
  </details>

<details><summary>&emsp;TuningReduction</summary>

  - **Inputs:**
    - tuning: ARRAY
  - **Outputs:**
    - reduced: ARRAY
  </details>

<details><summary>&emsp;VAMP</summary>

  - **Inputs:**
    - data: ARRAY
  - **Outputs:**
    - comps: ARRAY
  </details>

<details><summary>&emsp;VocalExpression</summary>

  - **Inputs:**
    - data: ARRAY
  - **Outputs:**
    - prosody_label: STRING
    - burst_label: STRING
    - prosody_score: ARRAY
    - burst_score: ARRAY
  </details>

<details><summary>&emsp;Walker</summary>

  - **Inputs:**
    - angle: ARRAY
    - velocity: ARRAY
    - water: ARRAY
  - **Outputs:**
    - latitude: ARRAY
    - longitude: ARRAY
  </details>

</details>

## Array

Nodes implementing array operations.

<details><summary>View Nodes</summary>

<details><summary>&emsp;Clip</summary>

  - **Inputs:**
    - array: ARRAY
  - **Outputs:**
    - out: ARRAY
  </details>

<details><summary>&emsp;Function</summary>

  - **Inputs:**
    - array: ARRAY
  - **Outputs:**
    - out: ARRAY
  </details>

<details><summary>&emsp;Join</summary>

  - **Inputs:**
    - a: ARRAY
    - b: ARRAY
  - **Outputs:**
    - out: ARRAY
  </details>

<details><summary>&emsp;Math</summary>

  - **Inputs:**
    - data: ARRAY
  - **Outputs:**
    - out: ARRAY
  </details>

<details><summary>&emsp;Operation</summary>

  - **Inputs:**
    - a: ARRAY
    - b: ARRAY
  - **Outputs:**
    - out: ARRAY
  </details>

<details><summary>&emsp;Reduce</summary>

  - **Inputs:**
    - array: ARRAY
  - **Outputs:**
    - out: ARRAY
  </details>

<details><summary>&emsp;Reshape</summary>

  - **Inputs:**
    - array: ARRAY
  - **Outputs:**
    - out: ARRAY
  </details>

<details><summary>&emsp;Select</summary>

  - **Inputs:**
    - data: ARRAY
  - **Outputs:**
    - out: ARRAY
  </details>

<details><summary>&emsp;Transpose</summary>

  - **Inputs:**
    - array: ARRAY
  - **Outputs:**
    - out: ARRAY
  </details>

</details>

## Inputs

Nodes that provide data to the pipeline.

<details><summary>View Nodes</summary>

<details><summary>&emsp;Audiocraft</summary>

  - **Inputs:**
    - prompt: STRING
  - **Outputs:**
    - wav: ARRAY
  </details>

<details><summary>&emsp;AudioStream</summary>

  - **Inputs:**
  - **Outputs:**
    - out: ARRAY
  </details>

<details><summary>&emsp;ConstantArray</summary>

  - **Inputs:**
  - **Outputs:**
    - out: ARRAY
  </details>

<details><summary>&emsp;ConstantString</summary>

  - **Inputs:**
  - **Outputs:**
    - out: STRING
  </details>

<details><summary>&emsp;ConstantTable</summary>

  - **Inputs:**
  - **Outputs:**
    - table: TABLE
  </details>

<details><summary>&emsp;EEGRecording</summary>

  - **Inputs:**
  - **Outputs:**
  </details>

<details><summary>&emsp;ExtendedTable</summary>

  - **Inputs:**
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
  - **Outputs:**
    - table: TABLE
  </details>

<details><summary>&emsp;FractalImage</summary>

  - **Inputs:**
    - complexity: ARRAY
  - **Outputs:**
    - image: ARRAY
  </details>

<details><summary>&emsp;ImageGeneration</summary>

  - **Inputs:**
    - prompt: STRING
    - negative_prompt: STRING
    - base_image: ARRAY
  - **Outputs:**
    - img: ARRAY
  </details>

<details><summary>&emsp;Kuramoto</summary>

  - **Inputs:**
    - initial_phases: ARRAY
  - **Outputs:**
    - phases: ARRAY
    - coupling: ARRAY
    - order_parameter: ARRAY
    - waveforms: ARRAY
  </details>

<details><summary>&emsp;LoadFile</summary>

  - **Inputs:**
    - file: STRING
  - **Outputs:**
    - data_output: ARRAY
    - string_output: STRING
  </details>

<details><summary>&emsp;LSLClient</summary>

  - **Inputs:**
    - source_name: STRING
    - stream_name: STRING
  - **Outputs:**
    - out: ARRAY
  </details>

<details><summary>&emsp;MeteoMedia</summary>

  - **Inputs:**
    - latitude: ARRAY
    - longitude: ARRAY
    - location_name: STRING
  - **Outputs:**
    - weather_data_table: TABLE
  </details>

<details><summary>&emsp;Oscillator</summary>

  - **Inputs:**
    - frequency: ARRAY
  - **Outputs:**
    - out: ARRAY
  </details>

<details><summary>&emsp;OSCIn</summary>

  - **Inputs:**
  - **Outputs:**
    - message: TABLE
  </details>

<details><summary>&emsp;PromptBook</summary>

  - **Inputs:**
    - input_prompt: STRING
  - **Outputs:**
    - out: STRING
  </details>

<details><summary>&emsp;RandomArray</summary>

  - **Inputs:**
  - **Outputs:**
    - random_array: ARRAY
  </details>

<details><summary>&emsp;Replay</summary>

  - **Inputs:**
  - **Outputs:**
    - table_output: TABLE
  </details>

<details><summary>&emsp;Reservoir</summary>

  - **Inputs:**
    - connectivity: ARRAY
  - **Outputs:**
    - data: ARRAY
  </details>

<details><summary>&emsp;SerialStream</summary>

  - **Inputs:**
  - **Outputs:**
    - out: ARRAY
  </details>

<details><summary>&emsp;SimulatedEEG</summary>

  - **Inputs:**
    - exponents: ARRAY
    - peaks: ARRAY
    - variances: ARRAY
    - peak_amplitudes: ARRAY
  - **Outputs:**
    - eeg_signal: ARRAY
  </details>

<details><summary>&emsp;SpikingNetwork</summary>

  - **Inputs:**
    - input: ARRAY
  - **Outputs:**
    - potentials: ARRAY
  </details>

<details><summary>&emsp;Table</summary>

  - **Inputs:**
    - base: TABLE
    - new_entry: ARRAY
  - **Outputs:**
    - table: TABLE
  </details>

<details><summary>&emsp;TextGeneration</summary>

  - **Inputs:**
    - prompt: STRING
  - **Outputs:**
    - generated_text: STRING
  </details>

<details><summary>&emsp;VectorDB</summary>

  - **Inputs:**
    - input_vector: ARRAY
  - **Outputs:**
    - top_labels: TABLE
    - vectors: ARRAY
  </details>

<details><summary>&emsp;VideoStream</summary>

  - **Inputs:**
  - **Outputs:**
    - frame: ARRAY
  </details>

<details><summary>&emsp;ZeroMQIn</summary>

  - **Inputs:**
  - **Outputs:**
    - data: ARRAY
  </details>

</details>

## Misc

Miscellaneous nodes that do not fit into other categories.

<details><summary>View Nodes</summary>

<details><summary>&emsp;AppendTables</summary>

  - **Inputs:**
    - table1: TABLE
    - table2: TABLE
  - **Outputs:**
    - output_table: TABLE
  </details>

<details><summary>&emsp;ArrayAwait</summary>

  - **Inputs:**
    - data: ARRAY
    - trigger: ARRAY
  - **Outputs:**
    - out: ARRAY
  </details>

<details><summary>&emsp;ColorEnhancer</summary>

  - **Inputs:**
    - image: ARRAY
  - **Outputs:**
    - enhanced_image: ARRAY
  </details>

<details><summary>&emsp;EdgeDetector</summary>

  - **Inputs:**
    - image: ARRAY
  - **Outputs:**
    - edges: ARRAY
  </details>

<details><summary>&emsp;FormatString</summary>

  - **Inputs:**
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
  - **Outputs:**
    - output_string: STRING
  </details>

<details><summary>&emsp;HSVtoRGB</summary>

  - **Inputs:**
    - hsv_image: ARRAY
  - **Outputs:**
    - rgb_image: ARRAY
  </details>

<details><summary>&emsp;JoinString</summary>

  - **Inputs:**
    - string1: STRING
    - string2: STRING
    - string3: STRING
    - string4: STRING
    - string5: STRING
  - **Outputs:**
    - output: STRING
  </details>

<details><summary>&emsp;LatentRotator</summary>

  - **Inputs:**
    - latent_vector: ARRAY
    - angles: ARRAY
  - **Outputs:**
    - rotated_vector: ARRAY
  </details>

<details><summary>&emsp;RGBtoHSV</summary>

  - **Inputs:**
    - rgb_image: ARRAY
  - **Outputs:**
    - hsv_image: ARRAY
  </details>

<details><summary>&emsp;SetMeta</summary>

  - **Inputs:**
    - array: ARRAY
  - **Outputs:**
    - out: ARRAY
  </details>

<details><summary>&emsp;StringAwait</summary>

  - **Inputs:**
    - message: STRING
    - trigger: ARRAY
  - **Outputs:**
    - out: STRING
  </details>

<details><summary>&emsp;StringToTable</summary>

  - **Inputs:**
    - text: STRING
  - **Outputs:**
    - table: TABLE
  </details>

<details><summary>&emsp;Switch</summary>

  - **Inputs:**
    - selector: ARRAY
    - array1: ARRAY
    - array2: ARRAY
    - array3: ARRAY
    - string1: STRING
    - string2: STRING
    - string3: STRING
  - **Outputs:**
    - array_out: ARRAY
    - string_out: STRING
  </details>

<details><summary>&emsp;TableSelectArray</summary>

  - **Inputs:**
    - input_table: TABLE
  - **Outputs:**
    - output_array: ARRAY
  </details>

<details><summary>&emsp;TableSelectString</summary>

  - **Inputs:**
    - input_table: TABLE
  - **Outputs:**
    - output_string: STRING
  </details>

<details><summary>&emsp;TableToString</summary>

  - **Inputs:**
    - table: TABLE
  - **Outputs:**
    - text: STRING
  </details>

</details>

## Outputs

Nodes that send data to external systems.

<details><summary>View Nodes</summary>

<details><summary>&emsp;AudioOut</summary>

  - **Inputs:**
    - data: ARRAY
    - device: STRING
  - **Outputs:**
    - finished: ARRAY
  </details>

<details><summary>&emsp;LSLOut</summary>

  - **Inputs:**
    - data: ARRAY
  - **Outputs:**
  </details>

<details><summary>&emsp;MidiCCout</summary>

  - **Inputs:**
    - cc1: ARRAY
    - cc2: ARRAY
    - cc3: ARRAY
    - cc4: ARRAY
    - cc5: ARRAY
  - **Outputs:**
    - midi_status: STRING
  </details>

<details><summary>&emsp;MidiOut</summary>

  - **Inputs:**
    - note: ARRAY
    - velocity: ARRAY
    - duration: ARRAY
  - **Outputs:**
    - midi_status: STRING
  </details>

<details><summary>&emsp;OSCOut</summary>

  - **Inputs:**
    - data: TABLE
  - **Outputs:**
  </details>

<details><summary>&emsp;SharedMemOut</summary>

  - **Inputs:**
    - data: ARRAY
  - **Outputs:**
  </details>

<details><summary>&emsp;WriteCsv</summary>

  - **Inputs:**
    - table_input: TABLE
    - start: ARRAY
    - stop: ARRAY
    - fname: STRING
  - **Outputs:**
  </details>

<details><summary>&emsp;ZeroMQOut</summary>

  - **Inputs:**
    - data: ARRAY
  - **Outputs:**
  </details>

</details>

## Signal

Nodes implementing signal processing operations.

<details><summary>View Nodes</summary>

<details><summary>&emsp;Autocorrelation</summary>

  - **Inputs:**
    - signal: ARRAY
  - **Outputs:**
    - autocorr: ARRAY
  </details>

<details><summary>&emsp;Buffer</summary>

  - **Inputs:**
    - val: ARRAY
  - **Outputs:**
    - out: ARRAY
  </details>

<details><summary>&emsp;BufferString</summary>

  - **Inputs:**
    - val: STRING
  - **Outputs:**
    - out: STRING
  </details>

<details><summary>&emsp;Cycle</summary>

  - **Inputs:**
    - signal: ARRAY
  - **Outputs:**
    - cycle: ARRAY
  </details>

<details><summary>&emsp;EEGHeadsetDetection</summary>

  - **Inputs:**
    - eeg_data: ARRAY
  - **Outputs:**
    - headset_status: ARRAY
  </details>

<details><summary>&emsp;EMD</summary>

  - **Inputs:**
    - data: ARRAY
  - **Outputs:**
    - IMFs: ARRAY
  </details>

<details><summary>&emsp;FFT</summary>

  - **Inputs:**
    - data: ARRAY
  - **Outputs:**
    - mag: ARRAY
    - phase: ARRAY
  </details>

<details><summary>&emsp;Filter</summary>

  - **Inputs:**
    - data: ARRAY
  - **Outputs:**
    - filtered_data: ARRAY
  </details>

<details><summary>&emsp;FOOOFaperiodic</summary>

  - **Inputs:**
    - psd_data: ARRAY
  - **Outputs:**
    - offset: ARRAY
    - exponent: ARRAY
    - cf_peaks: ARRAY
    - cleaned_psd: ARRAY
  </details>

<details><summary>&emsp;FrequencyShift</summary>

  - **Inputs:**
    - data: ARRAY
  - **Outputs:**
    - out: ARRAY
  </details>

<details><summary>&emsp;Hilbert</summary>

  - **Inputs:**
    - data: ARRAY
  - **Outputs:**
    - inst_amplitude: ARRAY
    - inst_phase: ARRAY
    - inst_frequency: ARRAY
  </details>

<details><summary>&emsp;IFFT</summary>

  - **Inputs:**
    - spectrum: ARRAY
    - phase: ARRAY
  - **Outputs:**
    - reconstructed: ARRAY
  </details>

<details><summary>&emsp;Normalization</summary>

  - **Inputs:**
    - data: ARRAY
  - **Outputs:**
    - normalized: ARRAY
  </details>

<details><summary>&emsp;PSD</summary>

  - **Inputs:**
    - data: ARRAY
  - **Outputs:**
    - psd: ARRAY
  </details>

<details><summary>&emsp;Recurrence</summary>

  - **Inputs:**
    - input_array: ARRAY
  - **Outputs:**
    - recurrence_matrix: ARRAY
    - RR: ARRAY
    - DET: ARRAY
    - LAM: ARRAY
  </details>

<details><summary>&emsp;Resample</summary>

  - **Inputs:**
    - data: ARRAY
  - **Outputs:**
    - out: ARRAY
  </details>

<details><summary>&emsp;ResampleJoint</summary>

  - **Inputs:**
    - data1: ARRAY
    - data2: ARRAY
  - **Outputs:**
    - out1: ARRAY
    - out2: ARRAY
  </details>

<details><summary>&emsp;Smooth</summary>

  - **Inputs:**
    - data: ARRAY
  - **Outputs:**
    - out: ARRAY
  </details>

<details><summary>&emsp;StaticBaseline</summary>

  - **Inputs:**
    - data: ARRAY
  - **Outputs:**
    - normalized: ARRAY
  </details>

<details><summary>&emsp;TableNormalization</summary>

  - **Inputs:**
    - table: TABLE
  - **Outputs:**
    - normalized: TABLE
  </details>

<details><summary>&emsp;Threshold</summary>

  - **Inputs:**
    - data: ARRAY
  - **Outputs:**
    - thresholded: ARRAY
  </details>

<details><summary>&emsp;TimeDelayEmbedding</summary>

  - **Inputs:**
    - input_array: ARRAY
  - **Outputs:**
    - embedded_array: ARRAY
  </details>

<details><summary>&emsp;WelfordsZTransform</summary>

  - **Inputs:**
    - data: ARRAY
  - **Outputs:**
    - normalized: ARRAY
  </details>

</details>
<!-- !!GOOFI_PIPE_NODE_LIST_END!! -->
