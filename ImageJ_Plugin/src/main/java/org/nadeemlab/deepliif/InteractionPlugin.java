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
    ImagePlus impOverlay;
    OverlayCreator overlayCreator;
    Scrollbar segThreshSlider, sizeGateSlider;


    public void run(String arg)
    {
        try
        {
            if (DeepliifClient.impID == 0) {
                UIHelper.showErrorDialog("Interactive adjustment can only be performed on a DeepLIIF result image.");
                return;
            }

            impOverlay = IJ.getImage();
            if (impOverlay == null || impOverlay.getID() != DeepliifClient.impID) {
                UIHelper.showErrorDialog("DeepLIIF result is not the active image.");
                return;
            }

            String filepathOriginal = FileHelper.concat(DeepliifClient.directory, DeepliifClient.name + "_Original.png");
            String filepathSeg = FileHelper.concat(DeepliifClient.directory, DeepliifClient.name + "_Seg.png");
            String filepathOverlay = FileHelper.concat(DeepliifClient.directory, DeepliifClient.name + "_SegOverlaid.png");

            ImagePlus impOrig = IJ.openImage(filepathOriginal);
            ImagePlus impSeg = IJ.openImage(filepathSeg);

            overlayCreator = new OverlayCreator(impOrig, impSeg, impOverlay);

            int segThresh = 80;
            int sizeThresh = 50;

            String prevScoring = FileHelper.readTextFile(FileHelper.concat(DeepliifClient.directory, DeepliifClient.name+"_scoring.json"));
            JSONObject json = new JSONObject(prevScoring);
            segThresh = json.getInt("prob_thresh");
            sizeThresh = json.getInt("size_thresh");

            int numTotal = json.getInt("num_total");
            int numPos = json.getInt("num_pos");
            double percentPos = json.getDouble("percent_pos");

            NonBlockingGenericDialog gd = new NonBlockingGenericDialog("DeepLIIF - Interactive Adjustment");
            gd.addSlider("Segmentation threshold", 0, 255, segThresh);
            gd.addSlider("Size gating", 0, 1024, sizeThresh);
            segThreshSlider = (Scrollbar)gd.getSliders().get(0);
            sizeGateSlider = (Scrollbar)gd.getSliders().get(1);
            gd.addDialogListener(this);
            gd.showDialog();

            if (gd.wasCanceled()) {
                updateOverlayFromFile(filepathOverlay);
                DeepliifClient.scoreDialog.updateMessage(ResultHandler.createScoreHtmlTable(numTotal, numPos, percentPos));
                return;
            }

            segThresh = segThreshSlider.getValue();
            sizeThresh = sizeGateSlider.getValue();
            DeepliifClient.scoreDialog.updateMessage("Updating scores, please wait ...");
            UIHelper.showStatusAndProgress("DeepLIIF processing ...", 0.3);

            String response = DeepliifClient.postprocess(filepathOriginal, filepathSeg, segThresh, sizeThresh);
            UIHelper.showStatusAndProgress("DeepLIIF saving files ...", 0.7);

            ResultHandler results = new ResultHandler(response);
            results.write(DeepliifClient.directory, DeepliifClient.name);
            UIHelper.showStatusAndProgress("DeepLIIF finished.", 1.0);

            DeepliifClient.scoreDialog.updateMessage(results.createScoreHtmlTable());
            updateOverlayFromFile(filepathOverlay);
        }
        catch (DeepliifException e) {
            UIHelper.showStatusAndProgress("DeepLIIF error.", 1.0);
            UIHelper.showErrorDialog(e.getMessage());
        }
        catch (Exception e) {
            UIHelper.showErrorDialog("An error has occurred.");
        }
    }


    private void updateOverlayFromFile(String filepathOverlay)
    {
        ImagePlus tempOverlay = IJ.openImage(filepathOverlay);
        int[] pixTemp = (int[])tempOverlay.getChannelProcessor().getPixels();
        int[] pixOverlay = (int[])impOverlay.getChannelProcessor().getPixels();
        if (pixOverlay.length != pixTemp.length)
            return;
        for (int i = 0; i < pixOverlay.length; ++i)
            pixOverlay[i] = pixTemp[i];
        impOverlay.updateAndDraw();
    }


    public boolean dialogItemChanged(GenericDialog gd, AWTEvent e)
    {
        DeepliifClient.scoreDialog.updateMessage("Preview only shown.<br><br>Click 'OK' to finalize<br>and update scores.");
        overlayCreator.update(segThreshSlider.getValue(), sizeGateSlider.getValue());
        return true;
    }
}