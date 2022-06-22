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


public class InferencePlugin implements PlugIn, WindowListener
{
    private static final int maxWidth = 3000;
    private static final int maxHeight = 3000;
    private static final String[] allowedResolutions = new String[]{"10x", "20x", "40x"};

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

            if (!checkImageSize(imp))
                return;
            ImagePlus impOrig = getImageSelection(imp);

            FileInfo info = imp.getOriginalFileInfo();
            if (info == null) {
                UIHelper.showErrorDialog("Image must be a file.");
                return;
            }
            String name = FileHelper.getBaseName(info.fileName);

            GenericDialog gd = new GenericDialog("DeepLIIF - Select Resolution");
            gd.addChoice("Resolution:", allowedResolutions, "20x");
 		    gd.showDialog();
            if (gd.wasCanceled())
                return;
            String resolution = gd.getNextChoice();

            String outputDirectory = createOutputDirectory(info.directory, name);
            String filepathOriginal = FileHelper.concat(outputDirectory, name + "_Original.png");
            IJ.saveAs(impOrig, "png", filepathOriginal);
            UIHelper.showStatusAndProgress("DeepLIIF processing ...", 0.3);

            String response = DeepliifClient.infer(filepathOriginal, resolution);
            UIHelper.showStatusAndProgress("DeepLIIF saving files ...", 0.7);

            ResultHandler results = new ResultHandler(response);
            results.write(outputDirectory, name);
            UIHelper.showStatusAndProgress("DeepLIIF finished", 1.0);

            ImagePlus impOverlay = IJ.openImage(FileHelper.concat(outputDirectory, name + "_SegOverlaid.png"));
            impOverlay = new ImagePlus(name + " - DeepLIIF Result", impOverlay.getChannelProcessor());
            overlayWindow = new ImageWindow(impOverlay);
            overlayWindow.addWindowListener(this);
            scoreDialog = new MutableHTMLDialog(name + " - DeepLIIF Scoring", results.createScoreHtmlTable(), false);
            scoreDialog.addWindowListener(this);

            DeepliifClient.impID = impOverlay.getID();
            DeepliifClient.name = name;
            DeepliifClient.directory = outputDirectory;
            DeepliifClient.scoreDialog = scoreDialog;

            DeepliifClient.roiNames = null;
            DeepliifClient.roiDirectories = null;
        }
        catch (DeepliifException e) {
            UIHelper.showStatusAndProgress("DeepLIIF error.", 1.0);
            UIHelper.showErrorDialog(e.getMessage());
        }
        catch (Exception e) {
            UIHelper.showErrorDialog("An error has occurred.");
        }
    }


    private boolean checkImageSize(ImagePlus imp)
    {
        if (imp.getRoi() == null) {
            if (imp.getWidth() <= maxWidth && imp.getHeight() <= maxHeight)
                return true;
            UIHelper.showErrorDialog("Image size is larger than " + maxWidth + " x " + maxHeight + " pixels.  Please select a smaller ROI.");
            return false;
        }
        Rectangle bounds = imp.getRoi().getBounds();
        if (bounds.getWidth() <= maxWidth && bounds.getHeight() <= maxHeight)
            return true;
        UIHelper.showErrorDialog("ROI size is larger than " + maxWidth + " x " + maxHeight + " pixels.  Please select a smaller ROI.");
        return false;
    }


    private ImagePlus getImageSelection(ImagePlus imp)
    {
        ImagePlus crop = imp.crop();
        if (imp.getRoi() != null && imp.getRoi().getMask() != null) {
            byte[] pixMask = (byte[])imp.getRoi().getMask().getPixels();
            int[] pixCrop = (int[])crop.getChannelProcessor().getPixels();
            for (int i = 0; i < pixMask.length; ++i)
                if (pixMask[i] == 0)
                    pixCrop[i] = 0xFFFFFFFF;
        }
        return crop;
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