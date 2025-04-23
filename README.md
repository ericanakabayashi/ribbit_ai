# Ribbit - Shazam for frogs!

The Ribbit web application was developed as a Capstone project for MIDS by Haissam Akhras,  Lia Cappellari, Farouk Ghandour, Erica Nakabayashi and Juliana Gómez Consuegra. 

## Objective

In a world where over 40% of amphibian species face extinction, partly due to the climate crisis, our app Ribbit is bridging critical data gaps in the Global South by empowering nature enthusiasts. Ribbit transforms your phone web browser into a tool for automated frog and toad identification. By simply recording a frog's call, our web app instantly analyzes the call and classifies the species, allowing users to contribute valuable data to global conservation efforts. 


## Methods
Following Ghani et al, 2023, we applied a linear probe on top of embeddings extracted using the BirdNet model developed by the Cornell Lab of Ornithology, to classify frogs and toads by using their calls. Our raw data included data from three sources: [iNat sounds](https://github.com/gvanhorn38/iNatSounds), [Anuraset](https://github.com/soundclim/anuraset) and [Anfibios del Ecuador](https://bioweb.bio/faunaweb/amphibiaweb/). 

## Folder structure [WIP]


## Running with uv (astral)

Raise venv using astral uv, activate, and sync dependencies
```
uv venv --python 3.11 
source .venv/bin/activate
uv sync
```