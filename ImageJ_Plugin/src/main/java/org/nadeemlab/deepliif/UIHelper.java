package org.nadeemlab.deepliif;

import ij.IJ;
import ij.gui.MessageDialog;


public class UIHelper
{
    private static final boolean debug = true;


    public static void log(String message)
    {
        if (debug)
            IJ.log(message);
    }


    public static void showErrorDialog(String message)
    {
        new MessageDialog(IJ.getInstance(), "Error", message);
    }


    public static void showStatusAndProgress(String status, double progress)
    {
        IJ.showStatus(status);
        IJ.showProgress(progress);
    }
}