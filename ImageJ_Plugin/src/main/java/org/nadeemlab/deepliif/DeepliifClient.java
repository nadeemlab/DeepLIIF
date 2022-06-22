package org.nadeemlab.deepliif;

import java.io.File;
import java.util.Vector;
import java.util.concurrent.TimeUnit;

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
    public static final int maxWidth = 3000;
    public static final int maxHeight = 3000;
    public static final String[] allowedResolutions = new String[]{"10x", "20x", "40x"};

    private static final OkHttpClient httpClient = new OkHttpClient.Builder().readTimeout(0, TimeUnit.SECONDS).build();


    public static String infer(String pathOriginal, String resolution) throws DeepliifException
    {
        String filename = FilenameUtils.getName(pathOriginal);

        HttpUrl url = new HttpUrl.Builder()
           .scheme("https")
           .host("deepliif.org")
           .addPathSegments("api/infer")
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
                code = response.code();
                body = response.body().string();
            }
        }
        catch (Exception e) {
            UIHelper.log("caught okhttp3 exception");
            UIHelper.log(e.toString());
            code = 0;
        }

        if (code != 200)
            throw new DeepliifException("Error communicating with DeepLIIF server.");
        return body;
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
            UIHelper.log("caught okhttp3 exception");
            UIHelper.log(e.toString());
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