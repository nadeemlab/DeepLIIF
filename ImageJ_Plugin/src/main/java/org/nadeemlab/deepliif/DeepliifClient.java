package org.nadeemlab.deepliif;

import java.io.File;
import java.util.Vector;
import java.util.concurrent.TimeUnit;

import org.json.JSONException;
import org.json.JSONObject;

import okhttp3.HttpUrl;
import okhttp3.MediaType;
import okhttp3.MultipartBody;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;
import okhttp3.Response;

import org.apache.commons.io.FilenameUtils;


public class DeepliifClient
{
    public static final int maxWidth = 4096;
    public static final int maxHeight = 4096;
    public static final String[] allowedResolutions = new String[]{"10x", "20x", "40x"};

    private static final OkHttpClient httpClient = new OkHttpClient.Builder().readTimeout(0, TimeUnit.SECONDS).build();


    public static String infer(String pathOriginal, String resolution) throws DeepliifException
    {
        try {
            String filename = FilenameUtils.getName(pathOriginal);

            HttpUrl url = new HttpUrl.Builder()
               .scheme("https")
               .host("deepliif.org")
               .addPathSegments("api/batch/infer")
               .addQueryParameter("resolution", resolution)
               .addQueryParameter("pil", "true")
               .build();

            RequestBody requestBody = new MultipartBody.Builder()
                .setType(MultipartBody.FORM)
                .addFormDataPart("img", filename, RequestBody.create(new File(pathOriginal), MediaType.parse("image/png")))
                .build();

            Request request = new Request.Builder()
                .url(url)
                .post(requestBody)
                .build();

            int code = 0;
            String body = null;

            try (Response response = httpClient.newCall(request).execute()) {
                if (response.isSuccessful()) {
                    body = response.body().string();
                    code = response.code();
                } else {
                    throw new DeepliifException("Error communicating with DeepLIIF server.");
                }
            }
            catch (Exception e) {
                throw new DeepliifException("Error communicating with DeepLIIF server.");
            }

            String taskID = (new JSONObject(body)).getString("task_id");
            
            url = new HttpUrl.Builder()
               .scheme("https")
               .host("deepliif.org")
               .addPathSegments("api/batch/infer/"+taskID)
               .build();

            request = new Request.Builder()
                .url(url)
                .build();

            JSONObject json = null;
            int countStatusTries = 0;

            while (code == 200)
            {
                ++countStatusTries;
                int waitTime = 1;
                if (countStatusTries > 24)
                    waitTime = 6;
                else if (countStatusTries > 22)
                    waitTime = 5;
                else if (countStatusTries > 19)
                    waitTime = 4;
                else if (countStatusTries > 15)
                    waitTime = 3;
                else if (countStatusTries > 10)
                    waitTime = 2;

                TimeUnit.SECONDS.sleep(waitTime);

                try (Response response = httpClient.newCall(request).execute()) {
                    if (response.isSuccessful()) {
                        body = response.body().string();
                        code = response.code();
                        json = new JSONObject(body);
                        String state = json.getString("state");
                        if (state.equals("SUCCESS"))
                            break;
                        else if (state.equals("FAILURE") || state.equals("REVOKED"))
                            throw new DeepliifException("Error communicating with DeepLIIF server.");
                    } else {
                        throw new DeepliifException("Error communicating with DeepLIIF server.");
                    }
                }
                catch (Exception e) {
                    throw new DeepliifException("Error communicating with DeepLIIF server.");
                }
            }

            if (code != 200)
                throw new DeepliifException("Error communicating with DeepLIIF server.");
            return json.getJSONObject("result").toString();
        }
        catch (Exception e) {
            throw new DeepliifException("Error communicating with DeepLIIF server.");
        }
    }


    public static String postprocess(String pathOriginal, String pathSeg, int segThresh, int sizeThresh) throws DeepliifException
    {
        String filenameOriginal = FilenameUtils.getName(pathOriginal);
        String filenameSeg = FilenameUtils.getName(pathSeg);

        HttpUrl url = new HttpUrl.Builder()
           .scheme("https")
           .host("deepliif.org")
           .addPathSegments("api/postprocess")
           .addQueryParameter("prob_thresh", ""+segThresh)
           .addQueryParameter("size_thresh", ""+sizeThresh)
           .build();

        RequestBody requestBody = new MultipartBody.Builder()
            .setType(MultipartBody.FORM)
            .addFormDataPart("img", filenameOriginal, RequestBody.create(new File(pathOriginal), MediaType.parse("image/png")))
            .addFormDataPart("seg_img", filenameSeg, RequestBody.create(new File(pathSeg), MediaType.parse("image/png")))
            .build();

        Request request = new Request.Builder()
            .url(url)
            .post(requestBody)
            .build();

        int code = 0;
        String body = null;

        try (Response response = httpClient.newCall(request).execute()) {
            if (response.isSuccessful()) {
                code = response.code();
                body = response.body().string();
            }
        }
        catch (Exception e) {
            code = 0;
        }

        if (code != 200)
            throw new DeepliifException("Error communicating with DeepLIIF server.");
        return body;
    }


    public static int impID;
    public static String name;
    public static String directory;
    public static MutableHTMLDialog scoreDialog;

    public static String[] roiNames;
    public static String[] roiDirectories;
    public static int[] roiOffsetsX;
    public static int[] roiOffsetsY;
}