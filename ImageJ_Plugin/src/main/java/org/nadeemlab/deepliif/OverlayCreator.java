package org.nadeemlab.deepliif;

import java.util.ArrayDeque;
import java.util.Vector;

import ij.ImagePlus;


public class OverlayCreator
{
    private ImagePlus impOverlay;
    private int width, height;
    int[] pixOrig, pixSeg, pixOverlay;
    byte[] posneg;


    OverlayCreator(ImagePlus orig, ImagePlus seg, ImagePlus overlay) throws DeepliifException
    {
        impOverlay = overlay;
        width = overlay.getWidth();
        height = overlay.getHeight();

        if (width != orig.getWidth() || width != seg.getWidth() || height != orig.getHeight() || height != seg.getHeight())
            throw new DeepliifException("Images do not have the same dimensions.");
        if (orig.getType() != ImagePlus.COLOR_RGB || seg.getType() != ImagePlus.COLOR_RGB || overlay.getType() != ImagePlus.COLOR_RGB)
            throw new DeepliifException("Images are not all of type RGB.");

        pixOrig = (int[])orig.getChannelProcessor().getPixels();
        pixSeg = (int[])seg.getChannelProcessor().getPixels();
        pixOverlay = (int[])overlay.getChannelProcessor().getPixels();
        posneg = new byte[pixOverlay.length];
    }


    public void update(int segThresh, int sizeThresh)
    {
        createPosNegMask(segThresh);
        computeCellMapping(sizeThresh);
        createOverlay();
        impOverlay.updateAndDraw();
    }


    private void createPosNegMask(int segThresh)
    {
        for (int i = 0; i < pixSeg.length; ++i) {
            int r = (pixSeg[i] >> 16) & 0xFF;
            int g = (pixSeg[i] >> 8) & 0xFF;
            int b = pixSeg[i] & 0xFF;
            if (r > segThresh && g <= 80 && b <= r)
                posneg[i] = 1;
            else if (b > segThresh && g <= 80 && r < b)
                posneg[i] = -1;
            else
                posneg[i] = 0;
        }
    }


    private void computeCellMapping(int sizeThresh)
    {
        ArrayDeque<Integer> xs = new ArrayDeque<Integer>();
        ArrayDeque<Integer> ys = new ArrayDeque<Integer>();
        Vector<Integer> cluster = new Vector<Integer>();

        int[] neigh_x = {-1, 0, 1, -1, 1, -1, 0, 1};
        int[] neigh_y = {-1, -1, -1, 0, 0, 1, 1, 1};

        for (int y = 0; y < height; ++y)
            for (int x = 0; x < width; ++x)
                if (posneg[y*width+x] == 1 || posneg[y*width+x] == -1)
                {
                    xs.add(x);
                    ys.add(y);
                    cluster.clear();
                    cluster.add(y*width+x);
                    int cluster_prob = posneg[y*width+x];
                    posneg[y*width+x] = 0;

                    while (!xs.isEmpty())
                    {
                        int x1 = xs.remove();
                        int y1 = ys.remove();

                        for (int n = 0; n < 8; ++n)
                        {
                            int nx = x1 + neigh_x[n];
                            int ny = y1 + neigh_y[n];
                            if (nx < 0 || nx >= width || ny < 0 || ny >= height)
                                continue;
                            if (posneg[ny*width+nx] == 1 || posneg[ny*width+nx] == -1) {
                                cluster.add(ny*width+nx);
                                xs.add(nx);
                                ys.add(ny);
                                cluster_prob += posneg[ny*width+nx];
                                posneg[ny*width+nx] = 0;
                            }
                        }
                    }

                    byte cluster_value = 0;
                    if (cluster.size() >= sizeThresh) {
                        if (cluster_prob < 0)
                            cluster_value = -2;
                        else
                            cluster_value = 2;
                    }
                    for (int i : cluster)
                        posneg[i] = cluster_value;
                }
    }


    private void createOverlay()
    {
        for (int i = 0; i < pixOrig.length; ++i)
            pixOverlay[i] = pixOrig[i];

        for (int y = 0, i = 0; y < height-1; ++y, ++i)
            for (int x = 0; x < width-1; ++x, ++i)
            {
                int sumX = posneg[i] + posneg[i+1];
                int sumY = posneg[i] + posneg[i+width];
                if (sumX == 2 || sumY == 2 || sumX == -2 || sumY == -2) {
                    int overlayColor = (sumX == 2 || sumY == 2) ? 0xFFFF0000 : 0xFF0000FF;
                    if (y > 0)
                        pixOverlay[i-width] = overlayColor;
                    if (x > 0)
                        pixOverlay[i-1] = overlayColor;
                    pixOverlay[i] = overlayColor;
                    pixOverlay[i+1] = overlayColor;
                    if (x < width-2)
                        pixOverlay[i+2] = overlayColor;
                    pixOverlay[i+width] = overlayColor;
                    if (y < height-2)
                        pixOverlay[i+width+width] = overlayColor;
                }
            }
    }
}