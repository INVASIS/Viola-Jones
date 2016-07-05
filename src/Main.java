import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;

/**
 * this is a main class
 */
public class Main {
    public static void main(String[] args) {
        loadLibraries();
    }

    public static ArrayList<File> listEndsWith(String path, String ext) {
        File dir = new File(path);
        File[] files = dir.listFiles((d, name) -> name.endsWith('.' + ext));
        return new ArrayList<>(Arrays.asList(files));
    }

    private static void loadLibraries() {
        String lib_ext;
        String path;
        if (System.getProperty("os.name").contains("Windows")) {
            path = "\\JCuda-All-0.7.5b-bin-windows-x86_64";
            lib_ext = "dll";
        } else {
            path = "/JCuda-All-0.7.5b-bin-linux-x86_64";
            lib_ext = "so";
        }

        ArrayList<File> files = new ArrayList<>();
        files.addAll(listEndsWith("libs" + path, lib_ext));
        for (File f : files) {
            System.out.println("Loading external library: " + f.getAbsolutePath());
            System.load(f.getAbsolutePath());
        }
    }
}
