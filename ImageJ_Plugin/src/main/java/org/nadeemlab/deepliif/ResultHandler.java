package org.nadeemlab.deepliif;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Base64;

import org.json.JSONException;
import org.json.JSONObject;


public class ResultHandler
{
    private JSONObject json;


    public ResultHandler(String results) throws DeepliifException
    {
        try {
            json = new JSONObject(results);
        }
        catch (JSONException e) {
            throw new DeepliifException("Error in parsing JSON.");
        }
    }


    public String getScoring() throws DeepliifException
    {
        try {
            return json.getJSONObject("scoring").toString(2);
        }
        catch (JSONException e) {
            throw new DeepliifException("Error in parsing JSON.");
        }
    }


    public String[] getImageNames() throws DeepliifException
    {
        try {
            return JSONObject.getNames(json.getJSONObject("images"));
        }
        catch (JSONException e) {
            throw new DeepliifException("Error in parsing JSON.");
        }
    }


    public String getEncodedImage(String name) throws DeepliifException
    {
        try {
            return json.getJSONObject("images").getString(name);
        }
        catch (JSONException e) {
            throw new DeepliifException("Error in parsing JSON.");
        }
    }


    public byte[] getDecodedImage(String name) throws DeepliifException
    {
        try {
            return Base64.getDecoder().decode(getEncodedImage(name));
        }
        catch (JSONException e) {
            throw new DeepliifException("Error in parsing JSON.");
        }
    }


    // Write scoring to JSON file and images to PNG files.
    public void write(String path, String name) throws DeepliifException
    {
        try {
            FileHelper.writeFile(getScoring(), path, name + "_scoring.json");
            String[] keys = getImageNames();
            for (String key : keys)
                FileHelper.writeFile(getDecodedImage(key), path, name + "_" + key + ".png");
        }
        catch (DeepliifException e) {
            throw e;
        }
        catch (Exception e) {
            throw new DeepliifException("Error in saving files.");
        }
    }


    public String createScoreHtmlTable()
    {
        String html = "<table>";
        try {
            JSONObject scoring = json.getJSONObject("scoring");
            if (scoring.has("num_total"))
                html += "<tr><td>Number of total nuclei:</td><td style=\"text-align: right\">" + scoring.getInt("num_total") + "</td></tr>";
            if (scoring.has("num_pos"))
                html += "<tr><td>Number of IHC+ cells:</td><td style=\"text-align: right\">" + scoring.getInt("num_pos") + "</td></tr>";
            if (scoring.has("percent_pos"))
                html += "<tr><td>Percentage of IHC+ cells:</td><td style=\"text-align: right\">" + scoring.getDouble("percent_pos") + " %</td></tr>";
        }
        catch (JSONException e) {}
        html += "</table>";
        return html;
    }


    public static String createScoreHtmlTable(int numTotal, int numPos, double percentPos)
    {
        String html = "<table>";
        html += "<tr><td>Number of total nuclei:</td><td style=\"text-align: right\">" + numTotal + "</td></tr>";
        html += "<tr><td>Number of IHC+ cells:</td><td style=\"text-align: right\">" + numPos + "</td></tr>";
        html += "<tr><td>Percentage of IHC+ cells:</td><td style=\"text-align: right\">" + percentPos + " %</td></tr>";
        html += "</table>";
        return html;
    }
}