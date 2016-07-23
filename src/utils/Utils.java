package utils;

import GUI.ImageHandler;
import process.Conf;
import utils.yield.Yielderable;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;

public class Utils {
    public static ArrayList<String> scanDir(String dir) {
        ArrayList<String> results = new ArrayList<>();
        try {
            Files.walk(Paths.get(dir)).forEach(filePath -> {
                if (Files.isRegularFile(filePath)) {
                    results.add(filePath.toString());
                }
            });
        } catch (IOException e) {
            e.printStackTrace();
        }
        return results;
    }

    public static ArrayList<BufferedImage> listImages(String dir) {
        ArrayList<BufferedImage> results = new ArrayList<>();
        for (String p : scanDir(dir)) {
            if (new File(p).isFile()) {
                try {
                    results.add(ImageIO.read(new File(p)));
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }
        return results;
    }

    public static Yielderable<BufferedImage> streamImages(String dir) {
        return yield -> {
            for (String p : scanDir(dir)) {
                if (new File(p).isFile()) {
                    try {
                        yield.returning(ImageIO.read(new File(p)));
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                }
            }
        };
    }

    public static Yielderable<ImageHandler> streamImageHandler(String dir) {
        return yield -> {
            for (String p : scanDir(dir)) {
                if (new File(p).isFile())
                    yield.returning(new ImageHandler(p));
            }
        };
    }

    public static int countFiles(String dir) {
        int count = 0;
        for (String p : scanDir(dir)) {
            if (new File(p).isFile())
                count++;
        }
        return count;
    }

    public static void computeHaar(File directory) {
        String[] fileList = directory.list();
        System.out.println(directory.getPath());
        for (int i = 0; i < fileList.length; i++) {
            String fileName = directory.getPath() + "/" + fileList[i];
            if (!fileList[i].endsWith(Conf.FEATURE_EXTENSION) && !Files.exists(Paths.get(fileName + Conf.FEATURE_EXTENSION))) {
                ImageHandler imageHandler = new ImageHandler(fileName);
                imageHandler.getFeatures();
            }
        }
    }



}
