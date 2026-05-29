# Ribbit — Shazam for Frogs 🐸

Ribbit is a citizen science tool that turns your browser into an automated frog and toad identifier. Record a frog call, and Ribbit classifies the species — helping fill critical biodiversity data gaps, especially across the Global South.

The project originated as a Capstone for UC Berkeley MIDS, built by Haissam Akhras, Lia Cappellari, Farouk Ghandour, Erica Nakabayashi, and Juliana Gómez Consuegra. It has since grown into an ongoing open project.

## Why it matters

Over 40% of amphibian species face extinction, yet acoustic monitoring data remains sparse in many regions. Ribbit empowers nature enthusiasts and researchers to contribute observations directly from the field, building a richer, community-driven dataset for conservation science.

## How it works

Following Ghani et al. (2023), we apply a linear probe on top of audio embeddings extracted with the [BirdNET model](https://github.com/kahst/BirdNET-Analyzer) from the Cornell Lab of Ornithology. Training data was drawn from three open sources:

- [iNat Sounds](https://github.com/gvanhorn38/iNatSounds)
- [Anuraset](https://github.com/soundclim/anuraset)
- [Anfibios del Ecuador](https://bioweb.bio/faunaweb/amphibiaweb/)

## Get involved

We welcome contributions of all kinds — new training data, species coverage, language translations, model improvements, or UI work. If you'd like to collaborate, reach out at **julianagc@ischool.berkeley.edu**. We'd love to hear from you.

## Happy Ribbiting! 🐸
