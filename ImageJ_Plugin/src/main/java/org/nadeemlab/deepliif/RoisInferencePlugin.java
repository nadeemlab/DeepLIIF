package org.nadeemlab.deepliif;

import java.awt.event.WindowEvent;
import java.awt.event.WindowListener;
import java.awt.Rectangle;

import ij.IJ;
import ij.ImagePlus;
import ij.WindowManager;
import ij.gui.GenericDialog;
import ij.gui.ImageWindow;
import ij.io.FileInfo;
import ij.plugin.PlugIn;
import ij.process.ColorProcessor;
import ij.process.ImageProcessor;

import ij.gui.Roi;
import ij.plugin.frame.RoiManager;


public class RoisInferencePlugin implements PlugIn, WindowListener
{
    ImageWindow overlayWindow = null;
    MutableHTMLDialog scoreDialog = null;


    public void run(String arg)
    {
        try
        {
            ImagePlus imp = IJ.getImage();
            if (imp == null) {
                IJ.noImage();
                return;
            }

            Roi[] rois = RoiManager.getRoiManager().getRoisAsArray();
            ImagePlus[] imps = getImageRoiSelections(imp, rois);
            imp.killRoi();

            FileInfo info = imp.getOriginalFileInfo();
            if (info == null) {
                UIHelper.showErrorDialog("Image must be a file.");
                return;
            }
            String name = FileHelper.getBaseName(info.fileName);

            ImagePlus resultFullImage = createResultBackgroundImage(imp, name + " - DeepLIIF ROIs", rois);
            for (int i = 0; i < rois.length; ++i)
                replaceImageRoi(resultFullImage, rois[i], imps[i]);

            overlayWindow = new ImageWindow(resultFullImage);

            GenericDialog gd = new GenericDialog("DeepLIIF - Select Resolution");
            gd.addChoice("Resolution:", DeepliifClient.allowedResolutions, "20x");
 		    gd.showDialog();
            if (gd.wasCanceled())
                return;
            String resolution = gd.getNextChoice();

            String outputDirectory = createOutputDirectory(info.directory, name);
            ImageWindow resultImageWindow = null;
            int count = 0;
            double progressStep = 0.9 / imps.length;
            double progress = 0.1;

            String[] resultFilepaths = new String[imps.length];
            String[] outputSubdirectories = new String[imps.length];
            String[] roiNames = new String[imps.length];

            int numTotal = 0;
            int numPos = 0;

            for (int i = 0; i < imps.length; ++i)
            {
                String roiName = imps[i].getTitle();
                outputSubdirectories[i] = createOutputSubdirectory(outputDirectory, roiName);
                roiName = name + "_" + roiName;
                String filepathOriginal = FileHelper.concat(outputSubdirectories[i], roiName + "_Original.png");
                IJ.saveAs(imps[i], "png", filepathOriginal);

                String message = "DeepLIIF processing " + (i+1) + "/" + imps.length + " ...";
                UIHelper.showStatusAndProgress(message, progress);

                String response = DeepliifClient.infer(filepathOriginal, resolution);
                ResultHandler results = new ResultHandler(response);
                results.write(outputSubdirectories[i], roiName);
                numTotal += results.getNumTotal();
                numPos += results.getNumPos();
                progress += progressStep;

                //ImagePlus resultImage = IJ.openImage(FileHelper.concat(outputSubdirectory, roiName + "_SegOverlaid.png"));
                //resultImage = new ImagePlus(roiName + " - DeepLIIF Result", resultImage.getChannelProcessor());
                //if (resultImageWindow == null)
                //    resultImageWindow = new ImageWindow(resultImage);
                //else
                //    resultImageWindow.setImage(resultImage);

                //resultFilepaths[i] = FileHelper.concat(outputSubdirectory, roiName + "_SegOverlaid.png");
                resultFilepaths[i] = FileHelper.concat(outputSubdirectories[i], roiName + "_SegOverlaid.png");
                roiNames[i] = roiName;
            }
            UIHelper.showStatusAndProgress("DeepLIIF finished", 1.0);

            for (int i = 0; i < resultFilepaths.length; ++i) {
                ImagePlus resultImage = IJ.openImage(resultFilepaths[i]);
                replaceImageRoi(resultFullImage, rois[i], resultImage);
            }
            resultFullImage.setTitle(name + " - DeepLIIF Results");
            resultFullImage.updateAndDraw();

            double percentPos = numPos * 100.0 / numTotal;
            scoreDialog = new MutableHTMLDialog(name + " - DeepLIIF Scoring", ResultHandler.createScoreHtmlTable(numTotal, numPos, percentPos), false);
            scoreDialog.addWindowListener(this);
            overlayWindow.addWindowListener(this);

            DeepliifClient.impID = resultFullImage.getID();
            DeepliifClient.name = name;
            DeepliifClient.directory = outputDirectory;
            DeepliifClient.scoreDialog = scoreDialog;

            DeepliifClient.roiNames = roiNames;
            DeepliifClient.roiDirectories = outputSubdirectories;

            int[] offsetsX = new int[rois.length];
            int[] offsetsY = new int[rois.length];
            for (int i = 0; i < rois.length; ++i) {
                offsetsX[i] = rois[i].getBounds().x;
                offsetsY[i] = rois[i].getBounds().y;
            }
            DeepliifClient.roiOffsetsX = offsetsX;
            DeepliifClient.roiOffsetsY = offsetsY;
        }
        catch (DeepliifException e) {
            UIHelper.showStatusAndProgress("DeepLIIF error.", 1.0);
            UIHelper.showErrorDialog(e.getMessage());
        }
        catch (Exception e) {
            UIHelper.showErrorDialog("An error has occurred.");
        }
    }


    private ImagePlus[] getImageRoiSelections(ImagePlus image, Roi[] rois) throws DeepliifException
    {
        if (rois.length == 0)
            throw new DeepliifException("No ROIs to process in ROI Manager.");

        for (Roi roi : rois) {
            Rectangle bounds = roi.getBounds();
            if (bounds.width > DeepliifClient.maxWidth || bounds.height > DeepliifClient.maxHeight)
                throw new DeepliifException("ROI is larger than " + DeepliifClient.maxWidth + " x " + DeepliifClient.maxHeight + " pixels.  Please reduce the ROI size.");
            if (bounds.x < 0 || bounds.y < 0 || bounds.x+bounds.width > image.getWidth() || bounds.y+bounds.height > image.getHeight())
                throw new DeepliifException("ROI boundary is outside of image.  All ROIs must be completely within the image.");
        }

        ImagePlus[] crops = image.crop(rois);
        for (ImagePlus crop : crops)
            maskOutsideRoi(crop, 0xFFFFFFFF);
        return crops;
    }


    private void maskOutsideRoi(ImagePlus imp, int color)
    {
        if (imp.getRoi() == null || imp.getRoi().getMask() == null)
            return;
        byte[] pixMask = (byte[])imp.getRoi().getMask().getPixels();
        int[] pixImage = (int[])imp.getChannelProcessor().getPixels();
        for (int i = 0; i < pixMask.length; ++i)
            if (pixMask[i] == 0)
                pixImage[i] = color;
    }


    private ImagePlus createResultBackgroundImage(ImagePlus image, String name, Roi[] rois)
    {
        ImagePlus result = image.duplicate();
        result.setTitle(name);

        int[] pixels = (int[])result.getChannelProcessor().getPixels();
        for (int i = 0; i < pixels.length; ++i)
            pixels[i] = 0xFE000000 | (pixels[i] & 0x00FFFFFF);

        ImageProcessor improc = result.getChannelProcessor();
        improc.setLineWidth(6);
        improc.setColor(0xFDFFFF00);
        for (int i = 0; i < rois.length; ++i)
            rois[i].drawPixels(improc);

        return result;
    }


    private void replaceImageRoi(ImagePlus image, Roi roi, ImagePlus roiImage)
    {
        ImageProcessor improcBkgnd = image.getChannelProcessor();
        ImageProcessor improcRoi = roiImage.getChannelProcessor();
        int offsetX = roi.getBounds().x;
        int offsetY = roi.getBounds().y;

        if (roi.getMask() == null) {
            for (int y = 0; y < improcRoi.getHeight(); ++y)
                for (int x = 0; x < improcRoi.getWidth(); ++x)
                    improcBkgnd.set(x+offsetX, y+offsetY, improcRoi.get(x, y));
        }
        else {
            ImageProcessor improcMask = roi.getMask();
            for (int y = 0; y < improcRoi.getHeight(); ++y)
                for (int x = 0; x < improcRoi.getWidth(); ++x)
                    if (improcMask.get(x, y) != 0)
                        improcBkgnd.set(x+offsetX, y+offsetY, improcRoi.get(x, y));
        }
    }


    private String createOutputDirectory(String parentDir, String name) throws DeepliifException
    {
        for (int i = 0; i < Integer.MAX_VALUE; ++i) {
            String dir = FileHelper.concat(parentDir, String.format("%s_DeepLIIF_%03d", name, i));
            if (FileHelper.notExists(dir)) {
                FileHelper.mkdirs(dir);
                return dir;
            }
        }
        throw new DeepliifException("Cannot create directory.");
    }


    private String createOutputSubdirectory(String directory, String roiName) throws DeepliifException
    {
        String dir = FileHelper.concat(directory, roiName);
        if (FileHelper.notExists(dir)) {
            FileHelper.mkdirs(dir);
            return dir;
        }
        throw new DeepliifException("Cannot create subdirectory.");
    }


    public void windowClosed(WindowEvent e)
    {
        if (overlayWindow == e.getWindow() && scoreDialog != null) {
            scoreDialog.dispose();
            overlayWindow = null;
        }
        else if (scoreDialog == e.getWindow() && overlayWindow != null) {
            overlayWindow.close();
            scoreDialog = null;
        }
    }

    public void windowActivated(WindowEvent e) {}
    public void windowClosing(WindowEvent e) {}
    public void windowDeactivated(WindowEvent e) {}
    public void windowDeiconified(WindowEvent e) {}
    public void windowIconified(WindowEvent e) {}
    public void windowOpened(WindowEvent e) {}
}