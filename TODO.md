## Improvements

### More feature types

The original paper only show us 5 different types of feature.
Actually, recent implementations use a lot more:

![Feature types](https://upload.wikimedia.org/wikipedia/commons/thumb/6/6e/Haar_features_Lienhart.svg/214px-Haar_features_Lienhart.svg.png)

Add them when possible


## Optimizations

### Reduce haar computing time by ~100x

Instead of sliding rectangles from size 1 x 1 to max x max on each image, 
we may only slide 100x less rectangles by providing to the algorithm an approx size of faces expected for that image.
This approach is compatible with existing CCTV cameras, so this would be a real improvement for performance.

Blacklist
ThreadPoolExecutor

