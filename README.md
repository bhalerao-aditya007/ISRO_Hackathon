---
title: PRISM API
emoji: 🌙
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
app_port: 7860
---
<div align="center">

<img src="https://upload.wikimedia.org/wikipedia/commons/b/bd/Indian_Space_Research_Organisation_Logo.svg" alt="ISRO Logo" width="150"/>

<br/>

# 🌌 PRISM

<img src="https://readme-typing-svg.herokuapp.com?font=Orbitron&size=26&duration=3000&pause=1000&color=FF9933&center=true&vCenter=true&width=800&lines=Multi-Agent+Lunar+Exploration+Framework;Harnessing+ISRO+Chandrayaan-2+Payloads;Generative+AI+for+Space+Data;Advanced+Crater+Detection+%26+Ice+Mapping" />

**An intelligent, multi-agent AI framework designed to process, analyze, and visualize multi-payload data from the Chandrayaan-2 mission.**

</div>

---

## 🎯 Problem Statement (PS)

**To design and develop an intelligent, multi-agent artificial intelligence framework capable of autonomously processing, fusing, and analyzing heterogeneous datasets from Chandrayaan-2 payloads to extract actionable planetary science insights.**

---

## 🚀 Overview

**PRISM** is an advanced Generative AI and Machine Learning pipeline engineered specifically for the Indian Space Research Organisation (ISRO). It tackles the extreme data sparsity of deep-space exploration by autonomously fusing physics-based datasets and utilizing cutting-edge deep learning to map the Moon's surface and exosphere.

By ingesting raw scientific products from Chandrayaan-2's diverse payloads, PRISM's multi-agent architecture performs high-resolution crater counting, subsurface ice classification, argon distribution mapping, and elemental abundance profiling.

---

## 🛰️ Chandrayaan-2 Payloads Integrated

<div align="center">

| Payload | Description | AI Agent Application |
| :---: | :--- | :--- |
| **DFSAR** | Dual-Frequency Synthetic Aperture Radar | Subsurface Ice & Regolith Classification |
| **OHRC** | Orbiter High-Resolution Camera | Self-Supervised Crater Detection & Topography |
| **ChACE-2** | Chandra's Atmospheric Composition Explorer | Exospheric Argon-40 Distribution |
| **CLASS** | Chandrayaan-2 Large Area Soft X-ray Spectrometer | Elemental Mapping (Mg, Al, Si, Ca, Ti, Fe) |
| **IIRS** | Imaging Infrared Spectrometer | Surface Thermal & Hydration Signatures |

</div>

---

## 🧠 Multi-Agent Architecture

PRISM relies on a distributed multi-agent system, where specialized AI models tackle discrete planetary science challenges. These agents feed into a unified backend, providing a holistic understanding of the lunar environment.

### 🌊 Generative AI Data Augmentation
To overcome the physical scarcity of labeled lunar radar data, PRISM employs a **Latent Diffusion UNet**. This neural network synthesizes physically-accurate synthetic SAR signatures based on genuine DFSAR data, enabling our classification agents to learn robust planetary boundaries without overfitting.

### 🌑 DeepMoon Crater CNN
Utilizing high-resolution **OHRC** imagery, our deep Convolutional Neural Network detects micro-craters using self-supervised morphological feature variance. This bypasses the need for manual human annotation while grounding the network entirely in genuine physical geometry.

---

## ⚙️ System Workflow

```mermaid
graph TD
    classDef isroOrange fill:#FF9933,stroke:#333,stroke-width:2px,color:black;
    classDef isroBlue fill:#000080,stroke:#333,stroke-width:2px,color:white;
    classDef isroGreen fill:#138808,stroke:#333,stroke-width:2px,color:white;

    subgraph ISRO Payloads
        DFSAR[DFSAR Data]:::isroOrange
        OHRC[OHRC Imagery]:::isroOrange
        ChACE2[ChACE-2 Scans]:::isroOrange
        CLASS[CLASS Spectra]:::isroOrange
        IIRS[IIRS Thermal]:::isroOrange
    end

    subgraph PRISM AI Core
        A1[Agent 1: PSR Subsurface Radar]:::isroBlue
        A2[Agent 2: Multi-Frequency SAR Ice]:::isroBlue
        A3[Agent 3: Exospheric Argon]:::isroBlue
        A4[Agent 4: DeepMoon CNN Craters]:::isroBlue
        A5[Agent 5: Elemental Mapping]:::isroBlue
        A6[Agent 6: Thermal/Hydration]:::isroBlue
    end

    subgraph Deep Generative Engine
        LD[PyTorch Latent Diffusion]:::isroGreen
        LD -.-> |Synthetic Signatures| A2
    end

    DFSAR --> A1
    DFSAR --> A2
    ChACE2 --> A3
    OHRC --> A4
    CLASS --> A5
    IIRS --> A6

    subgraph Visualization Interface
        API[FastAPI Gateway]:::isroBlue
        UI[Interactive Web Dashboard]:::isroOrange
    end

    A1 & A2 & A3 & A4 & A5 & A6 --> API
    API --> UI
```

---

## 🛠️ Technology Stack

* **Machine Learning:** PyTorch, Scikit-Learn, Pandas, NumPy
* **Generative AI:** Latent Diffusion Models, Convolutional Neural Networks
* **Backend Integration:** FastAPI, Uvicorn, Python 3.11+
* **Data Processing:** Rasterio, GeoPandas, Pillow
* **Architecture:** Multi-Agent AI Framework

<div align="center">
  <br/>
  <b>"Exploring the Moon, one pixel at a time."</b>
  <br/><br/>
  <img src="https://upload.wikimedia.org/wikipedia/commons/b/bd/Indian_Space_Research_Organisation_Logo.svg" alt="ISRO Logo" width="50"/>
</div>
