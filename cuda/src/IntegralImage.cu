extern "C"
__global__ void integral_image(int** src, int height, int width, int** dst)
{
    int x, y, s, t;
    for(y = 0; y < height; y++) {
        s = 0;
        /* loop over the number of columns */
        for(x = 0; x < width; x ++) {
            /* sum of the current row (integer)*/
            s += src[y][x];
            t = s;
            if (y != 0) {
	            t += dst[y-1][x];
	        }
	        dst[y][x] = t;
	    }
	}
}