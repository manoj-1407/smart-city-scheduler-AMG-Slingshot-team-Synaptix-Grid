# 🏙️ Adaptive AI Workload Scheduler for Smart Cities

> Real-time simulation of edge–fog–cloud task distribution across three scheduling strategies — built for the **AMD Slingshot Hackathon | AI for Smart Cities** track.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.38+-red?style=flat-square&logo=streamlit)
![Plotly](https://img.shields.io/badge/Plotly-5.22+-purple?style=flat-square&logo=plotly)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![AMD](https://img.shields.io/badge/AMD-Slingshot%20Hackathon-ED1C24?style=flat-square)

---

## 📌 Table of Contents

1. [Problem Statement](#-problem-statement)
2. [Solution Overview](#-solution-overview)
3. [Algorithm](#-adaptive-scheduling-algorithm)
4. [Features](#-features)
5. [Tech Stack](#-tech-stack)
6. [Getting Started](#-getting-started)
7. [How to Use](#-how-to-use-the-simulator)
8. [Example Scenarios](#-example-scenarios)
9. [Architecture](#-architecture)
10. [Project Structure](#-project-structure)
11. [Results](#-benchmark-results)
12. [AMD Integration](#-amd-integration)
13. [Roadmap](#-roadmap)
14. [Team](#-team)
15. [License](#-license)

---

## 🔴 Problem Statement

Smart cities like **Bengaluru, Dubai, Singapore, and Mumbai** generate thousands of concurrent AI tasks per second:

- 🚦 Traffic cameras — congestion detection, signal optimization, license plate recognition  
- 🌫️ IoT sensors — air quality, noise pollution, water quality monitoring  
- 🚨 Emergency dispatch — accident detection, ambulance routing, fire alerts  
- 📷 Public safety CCTV — crowd monitoring, anomaly detection, trespassing alerts  

These tasks must be distributed across **heterogeneous compute infrastructure**:

| Tier | Location | Capacity | Latency |
|------|----------|----------|---------|
| **Edge** | Traffic junctions, roadside units | Low (300 tasks) | Low (0.8) |
| **Fog** | District data centres | Medium (500 tasks) | Medium (0.4) |
| **Cloud** | Regional cloud (AMD-powered) | High (800–1000 tasks) | Low (0.2) |

**Current approaches fail:**

- **Static Scheduling** (always send to one node) → edge servers crash, cloud sits idle
- **Round Robin** (cycle blindly) → ignores capacity differences, unfair load distribution
- **Manual tuning** → reactive, fragile, creates configuration drift

**Real consequences:**
- Emergency SLAs missed — ambulance dispatch analytics delayed 8–12 seconds on saturated edge nodes  
- 60–70% of cloud capacity wasted while edge nodes burn out  
- During burst events (Diwali traffic, accidents, storms) — latency spikes 4–10×, SLA breaches hit 89%

---

## 💡 Solution Overview

An **Adaptive Cost-Based Workload Scheduler** that:

1. Models a realistic **edge–fog–cloud topology** with heterogeneous nodes  
2. Generates synthetic but realistic smart-city AI tasks with type, workload, and priority  
3. Benchmarks **3 scheduling strategies** on identical workloads  
4. Tracks **5 key metrics** per scheduler: latency, overloads, SLA breach, throughput, fairness  
5. Presents everything through an interactive **Streamlit** web app with **7+ Plotly charts**

City engineers can **tune parameters (penalty, SLA threshold, nodes, task count, burst mode)** and instantly see the impact — before touching production.

---

## ⚙️ Adaptive Scheduling Algorithm

For every incoming task `t` and every candidate node `n`, the Adaptive scheduler computes:

**Step 1 — Task Latency on this node:**
