# Overview

This repository contains the code and resources for a comparative analysis of the robustness of four popular machine learning models—Logistic Regression, Support Vector Machine (SVM), Random Forest, and Gradient Boosting—against various adversarial attack methods. The study evaluates the performance of these models when subjected to adversarial attacks such as FGSM, DeepFool, Carlini & Wagner (C&W), and Zero Order Optimization (ZOO) using the Adversarial Robustness Toolbox (ART).
Contents

A COMPARATIVE ANALYSIS OF ADVERSARIAL ATTACK METHODS USING MACHINE LEARNING
Key Findings

    Logistic Regression and SVM showed significant vulnerabilities to FGSM attacks but better resilience against more complex attacks like DeepFool, C&W, and ZOO.
    Random Forest and Gradient Boosting demonstrated higher robustness overall but faced challenges with advanced attacks like C&W.
    Execution time it was varied significantly, with FGSM being the fastest and ZOO the most time-consuming.

Future Work

    Explore advanced defense mechanisms such as adversarial training, defensive distillation, and ensemble modeling to further enhance model robustness.
    Develop resilient simulation tools to mimic real-world adversarial scenarios better.
    Investigate interdisciplinary approaches combining cybersecurity, human-computer interaction, and ethics for comprehensive defense strategies.
