# People Counting System - Public Demo

## Overview
This is a **simplified demonstration** of a real-time people counting system concept. It showcases the core architecture and functionality using pre-recorded video files and YOLOX object detection.

**Purpose**: Educational demo to visualize main ideas of the production system without exposing proprietary code or sensitive information.

## 🚀 Demo Scope

### What This Demo Shows
- YOLOX-based people detection
- Multi-object tracking with unique IDs
- Line-crossing counting logic (in/out)
- Multi-video processing simulation
- Real-time visualization interface
- Statistics aggregation and export

### What This Demo Does NOT Include
- NPU edge optimization (uses CPU/GPU)
- Live camera streaming (uses video files)
- Production-grade shared memory IPC
- Custom-trained detection models
- GenICam SDK integration
- Database persistence layer

## 🔧 Tech Stack

### Demo Technologies
- **Language**: Python 3.8+
- **Object Detection**: YOLOX (pre-trained on COCO dataset)
- **Video Processing**: OpenCV
- **Tracking**: Centroid-based tracker or norfair
- **Visualization**: OpenCV GUI / Streamlit
- **Data Export**: JSON/CSV

## 📊 Key Metrics (Production System Reference)

The actual production system achieves:
- **Throughput**: 4+ simultaneous 1080p/15fps streams
- **Accuracy**: 95%+ in adverse conditions
- **Latency**: <50ms per frame
- **Architecture**: Multi-device NPU edge deployment

This demo approximates the workflow but runs on standard hardware.


## 🎯 Limitations

1. **Not Real-Time Streaming**: Uses pre-recorded videos
2. **No Hardware Acceleration**: Runs on CPU/GPU, not NPU
3. **Simplified Architecture**: Threading instead of production IPC
4. **Basic Tracking**: Simple algorithm vs. advanced multi-object tracking
5. **No Persistence**: Statistics are exported to files, not database

## 📄 License

This demo is for educational purposes. The production system is proprietary.

## 🙏 Acknowledgments

This demo was created to showcase concepts from a real-world deployment without exposing sensitive IP or violating NDAs.