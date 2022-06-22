package org.nadeemlab.deepliif;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;

import org.apache.commons.io.FilenameUtils;


public class FileHelper
{
    public static String concat(String basePath, String appendPath)
    {
        return FilenameUtils.concat(basePath, appendPath);
    }


    // Get the base name (without extension) from a filepath.
    public static String getBaseName(String filepath)
    {
        return FilenameUtils.getBaseName(filepath);
    }


    // Get the name (with extension) from a filepath.
    public static String getName(String filepath)
    {
        return FilenameUtils.getName(filepath);
    }


    public static boolean notExists(String path) throws DeepliifException
    {
        try {
            return Files.notExists(Paths.get(path));
        }
        catch (Exception e) {
            throw new DeepliifException("Error in accessing filesystem.");
        }
    }


    public static void mkdirs(String path) throws DeepliifException
    {
        try {
            File directory = new File(path);
            directory.mkdirs();
            if (!directory.isDirectory())
                throw new DeepliifException("Error in creating directory.");
        }
        catch (Exception e) {
            throw new DeepliifException("Error in creating directory.");
        }
    }


    public static String readTextFile(String path) throws IOException
    {
        return new String(Files.readAllBytes(Paths.get(path)));
    }


    public static void writeFile(byte[] data, String path, String filename) throws FileNotFoundException, IOException
    {
        FileOutputStream stream = new FileOutputStream(FilenameUtils.concat(path, filename));
        stream.write(data);
        stream.close();
    }


    public static void writeFile(String str, String path, String filename) throws IOException
    {
        FileWriter writer = new FileWriter(FilenameUtils.concat(path, filename));
        writer.write(str);
        writer.close();
    }
}