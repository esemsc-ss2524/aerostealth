# Composing a CFD adjoint and an EM adjoint for gradient-based stealth vs aero co-design

## Abstract

## 1. Problem

## 2. Method: four Tesseracts behind one jvp/vjp interface

### 2.1 geom (reverse-mode autodiff)

### 2.2 cfd (continuous adjoint)

### 2.3 em (discrete FDTD adjoint)

### 2.4 optimize (tesseract-jax composition, epsilon-constraint sweep)

## 3. Results

### 3.1 Gradient agreement (per-Tesseract and end-to-end vs finite differences)

### 3.2 Pareto front: baseline vs selected designs

### 3.3 Adjoint vs finite-difference cost scaling

## 4. Discussion

## 5. Reproducibility
