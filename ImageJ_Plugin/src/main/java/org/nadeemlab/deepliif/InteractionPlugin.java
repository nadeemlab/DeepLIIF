package org.nadeemlab.deepliif;

import java.awt.AWTEvent;
import java.awt.Scrollbar;
import java.io.File;

import ij.IJ;
import ij.ImagePlus;
import ij.gui.DialogListener;
import ij.gui.GenericDialog;
import ij.gui.ImageWindow;
import ij.gui.NonBlockingGenericDialog;
import ij.plugin.PlugIn;
import ij.process.ImageProcessor;

import okhttp3.HttpUrl;
import okhttp3.MediaType;
import okhttp3.MultipartBody;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;
import okhttp3.Response;

import org.json.JSONObject;


public class InteractionPlugin implements PlugIn, DialogListener
{
    ImagePlus impOverlaid;
    ImagePlus[] impOverCreated;
    OverlayCreator[] overlayCreator;
    Scrollbar segThreshSlider, sizeGateSlider;


    public void run(String arg)
    {
        try
        {
            if (DeepliifClient.impID == 0) {
                UIHelper.showErrorDialog("Interactive adjustment can only be performed on a DeepLIIF result image.");
                return;
            }

            impOverlaid = IJ.getImage();
            if (impOverlaid == null || impOverlaid.getID() != DeepliifClient.impID) {
                UIHelper.showErrorDialog("DeepLIIF result is not the active image.");
                return;
            }

            final boolean hasRois = (DeepliifClient.roiNames != null) ? true : false;
            final int numImages = (hasRois) ? DeepliifClient.roiNames.length : 1;

            overlayCreator = new OverlayCreator[numImages];
            impOverCreated = new ImagePlus[numImages];
            String[] filepathsOriginal = new String[numImages];
            String[] filepathsSeg = new String[numImages];
            String[] filepathsOverlaid = new String[numImages];

            int segThresh = 80;
            int sizeThresh = 50;
            int numTotal = 0, numPos = 0;

            if (hasRois)
            {
                for (int i = 0; i < DeepliifClient.roiDirectories.length; ++i)
                {
                    //UIHelper.log(DeepliifClient.roiDirectories[i] + "  " + DeepliifClient.roiNames[i] + " " + DeepliifClient.roiOffsetsX[i] + "," + DeepliifClient.roiOffsetsY[i]);
                    filepathsOriginal[i] = FileHelper.concat(DeepliifClient.roiDirectories[i], DeepliifClient.roiNames[i] + "_Original.png");
                    filepathsSeg[i] = FileHelper.concat(DeepliifClient.roiDirectories[i], DeepliifClient.roiNames[i] + "_Seg.png");
                    filepathsOverlaid[i] = FileHelper.concat(DeepliifClient.roiDirectories[i], DeepliifClient.roiNames[i] + "_SegOverlaid.png");

                    ImagePlus impOrig = IJ.openImage(filepathsOriginal[i]);
                    ImagePlus impSeg = IJ.openImage(filepathsSeg[i]);
                    impOverCreated[i] = IJ.openImage(filepathsOverlaid[i]);
                    overlayCreator[i] = new OverlayCreator(impOrig, impSeg, impOverCreated[i]);

                    String prevScoring = FileHelper.readTextFile(FileHelper.concat(DeepliifClient.roiDirectories[i], DeepliifClient.roiNames[i]+"_scoring.json"));
                    JSONObject json = new JSONObject(prevScoring);
                    segThresh = json.getInt("prob_thresh");
                    sizeThresh = json.getInt("size_thresh");
                    numTotal += json.getInt("num_total");
                    numPos += json.getInt("num_pos");
                }
            }
            else
            {
                filepathsOriginal[0] = FileHelper.concat(DeepliifClient.directory, DeepliifClient.name + "_Original.png");
                filepathsSeg[0] = FileHelper.concat(DeepliifClient.directory, DeepliifClient.name + "_Seg.png");
                filepathsOverlaid[0] = FileHelper.concat(DeepliifClient.directory, DeepliifClient.name + "_SegOverlaid.png");

                ImagePlus impOrig = IJ.openImage(filepathsOriginal[0]);
                ImagePlus impSeg = IJ.openImage(filepathsSeg[0]);
                overlayCreator[0] = new OverlayCreator(impOrig, impSeg, impOverlaid);

                String prevScoring = FileHelper.readTextFile(FileHelper.concat(DeepliifClient.directory, DeepliifClient.name+"_scoring.json"));
                JSONObject json = new JSONObject(prevScoring);
                segThresh = json.getInt("prob_thresh");
                sizeThresh = json.getInt("size_thresh");
                numTotal = json.getInt("num_total");
                numPos = json.getInt("num_pos");
            }

            NonBlockingGenericDialog gd = new NonBlockingGenericDialog("DeepLIIF - Interactive Adjustment");
            gd.addSlider("Segmentation threshold", 0, 255, segThresh);
            gd.addSlider("Size gating", 0, 1024, sizeThresh);
            segThreshSlider = (Scrollbar)gd.getSliders().get(0);
            sizeGateSlider = (Scrollbar)gd.getSliders().get(1);
            gd.addDialogListener(this);
            gd.showDialog();

            if (gd.wasCanceled()) {
                if (hasRois)
                    updateOverlaidFromFiles(filepathsOverlaid);
                else
                    updateOverlaidFromFile(filepathsOverlaid[0]);
                double percentPos = numPos * 100.0 / numTotal;
                DeepliifClient.scoreDialog.updateMessage(ResultHandler.createScoreHtmlTable(numTotal, numPos, percentPos));
                return;
            }

            segThresh = segThreshSlider.getValue();
            sizeThresh = sizeGateSlider.getValue();

            if (hasRois)
            {
                numTotal = 0;
                numPos = 0;
                double progressStep = 0.9 / numImages;
                double progress = 0.1;

                for (int i = 0; i < numImages; ++i)
                {
                    String message = "DeepLIIF processing " + (i+1) + "/" + numImages + " ...";
                    UIHelper.showStatusAndProgress(message, progress);

                    String response = DeepliifClient.postprocess(filepathsOriginal[i], filepathsSeg[i], segThresh, sizeThresh);
                    ResultHandler results = new ResultHandler(response);
                    results.write(DeepliifClient.roiDirectories[i], DeepliifClient.roiNames[i]);
                    numTotal += results.getNumTotal();
                    numPos += results.getNumPos();
                    progress += progressStep;
                }

                UIHelper.showStatusAndProgress("DeepLIIF finished.", 1.0);

                double percentPos = numPos * 100.0 / numTotal;
                DeepliifClient.scoreDialog.updateMessage(ResultHandler.createScoreHtmlTable(numTotal, numPos, percentPos));
                updateOverlaidFromFiles(filepathsOverlaid);
            }
            else
            {
                DeepliifClient.scoreDialog.updateMessage("Updating scores, please wait ...");
                UIHelper.showStatusAndProgress("DeepLIIF processing ...", 0.3);

                String response = DeepliifClient.postprocess(filepathsOriginal[0], filepathsSeg[0], segThresh, sizeThresh);
                UIHelper.showStatusAndProgress("DeepLIIF saving files ...", 0.7);

                ResultHandler results = new ResultHandler(response);
                results.write(DeepliifClient.directory, DeepliifClient.name);
                UIHelper.showStatusAndProgress("DeepLIIF finished.", 1.0);

                DeepliifClient.scoreDialog.updateMessage(results.createScoreHtmlTable());
                updateOverlaidFromFile(filepathsOverlaid[0]);
            }
        }
        catch (DeepliifException e) {
            UIHelper.showStatusAndProgress("DeepLIIF error.", 1.0);
            UIHelper.showErrorDialog(e.getMessage());
        }
        catch (Exception e) {
            UIHelper.showErrorDialog("An error has occurred.");
        }
    }


    private void updateOverlaidFromFile(String filepathOverlaid)
    {
        ImagePlus tempOverlaid = IJ.openImage(filepathOverlaid);
        int[] pixTemp = (int[])tempOverlaid.getChannelProcessor().getPixels();
        int[] pixOverlaid = (int[])impOverlaid.getChannelProcessor().getPixels();
        if (pixOverlaid.length != pixTemp.length)
            return;
        for (int i = 0; i < pixOverlaid.length; ++i)
            pixOverlaid[i] = pixTemp[i];
        impOverlaid.updateAndDraw();
    }


    private void updateOverlaidFromFiles(String[] filespathsOverlaid)
    {
        ImageProcessor resultImage = impOverlaid.getChannelProcessor();
        for (int i = 0; i < filespathsOverlaid.length; ++i) {
            ImagePlus tempOverlaid = IJ.openImage(filespathsOverlaid[i]);
            ImageProcessor overlaidImage = tempOverlaid.getChannelProcessor();
            int offsetX = DeepliifClient.roiOffsetsX[i];
            int offsetY = DeepliifClient.roiOffsetsY[i];
            for (int y = 0; y < overlaidImage.getHeight(); ++y)
                for (int x = 0; x < overlaidImage.getWidth(); ++x)
                    if ((resultImage.get(x+offsetX, y+offsetY) & 0xFF000000) == 0xFF000000)
                        resultImage.set(x+offsetX, y+offsetY, overlaidImage.get(x, y));
        }
        impOverlaid.updateAndDraw();
    }


    public boolean dialogItemChanged(GenericDialog gd, AWTEvent e)
    {
        DeepliifClient.scoreDialog.updateMessage("Preview only shown.<br><br>Click 'OK' to finalize<br>and update scores.");
        for (int i = 0; i < overlayCreator.length; ++i)
            overlayCreator[i].update(segThreshSlider.getValue(), sizeGateSlider.getValue());

        if (DeepliifClient.roiNames != null) {
            ImageProcessor resultImage = impOverlaid.getChannelProcessor();
            for (int i = 0; i < impOverCreated.length; ++i) {
                ImageProcessor overlaidImage = impOverCreated[i].getChannelProcessor();
                int offsetX = DeepliifClient.roiOffsetsX[i];
                int offsetY = DeepliifClient.roiOffsetsY[i];
                for (int y = 0; y < overlaidImage.getHeight(); ++y)
                    for (int x = 0; x < overlaidImage.getWidth(); ++x)
                        if ((resultImage.get(x+offsetX, y+offsetY) & 0xFF000000) == 0xFF000000)
                            resultImage.set(x+offsetX, y+offsetY, overlaidImage.get(x, y));
            }
        }

        impOverlaid.updateAndDraw();
        return true;
    }
}