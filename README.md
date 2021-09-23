# csd3

## Perspective transformation
verwacht: camera input, perspective points, output resolutie
output: frame

## Feature detection
verwacht: frame
output: coordinaten, percentage black pixels, centre point

## Image generation
verwacht: feature detection output (background subtraction)
output: frame

## OSC module
verwacht: data output van feature detection & image generation 
output: OSC messages