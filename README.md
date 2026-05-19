# Real-Time MLB Win Probability Prediction System

This repository provides the HTML-based implementation of a real-time MLB win probability prediction dashboard developed for the study:

"Design and Implementation of a Real-Time MLB Game Outcome Prediction System Using Deep Learning and the MLB Stats API"

## Overview

This system uses the MLB Stats API to collect game information and visualizes pre-game and in-game win probabilities. The trained DNN model weights are embedded in JSON format and inference is performed in the browser using JavaScript matrix operations.

## Main Features

- Pre-game win probability prediction
- In-game win probability update by inning
- MLB Stats API integration
- Client-side DNN inference using JavaScript
- No dedicated server required

## Files

- `index.html`: Main dashboard file
- `dashboard_example.png`: Example screenshot of the implemented system

## How to Use

1. Download or clone this repository.
2. Open `index.html` in a web browser.
3. The dashboard will request data from the MLB Stats API and display win probability results.

## Notes

The system was trained using 2025 MLB regular-season data. API responses may vary depending on MLB Stats API availability and schedule status.
Web demo: https://sportintelligence.github.io/mlb-win-probability-system/
