package utils;


import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;
import org.xml.sax.SAXException;
import process.Conf;
import process.StumpRule;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.parsers.ParserConfigurationException;
import javax.xml.transform.Transformer;
import javax.xml.transform.TransformerConfigurationException;
import javax.xml.transform.TransformerException;
import javax.xml.transform.TransformerFactory;
import javax.xml.transform.dom.DOMSource;
import javax.xml.transform.stream.StreamResult;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;

import static utils.Utils.currentDate;
import static utils.Utils.fileExists;

public class CascadeSerializer {

    private static TransformerFactory transformerFactory;
    private static Transformer transformer;
    private static DOMSource source;
    private static StreamResult resultFile;
    private static boolean init = initXML();
    private static DocumentBuilderFactory factory;
    private static DocumentBuilder builder;
    private static Document document;
    private static Element root;

    private static boolean initXML() {
        try {
            factory = DocumentBuilderFactory.newInstance();
            builder = factory.newDocumentBuilder();
            document = builder.newDocument();
            root = document.createElement("Cascade");
            document.appendChild(root);

            transformerFactory = TransformerFactory.newInstance();
            transformer = transformerFactory.newTransformer();
            resultFile = new StreamResult(new File(Conf.TRAIN_DIR + "/cascade-" + currentDate(null) + ".data"));
        } catch (ParserConfigurationException | TransformerConfigurationException e) {
            e.printStackTrace();
        }
        return true;
    }

    private static void updateFile() {
        source = new DOMSource(document);
        try {
            transformer.transform(source, resultFile);
        } catch (TransformerException e) {
            e.printStackTrace();
        }
    }

    public static void writeCascadeLayerToXML(int round, ArrayList<StumpRule> committee, Float tweakValue) {
        boolean init = CascadeSerializer.init;

        Element layer = document.createElement("Layer");
        layer.setAttribute("id", String.valueOf(round));
        root.appendChild(layer);

        Element tweak = document.createElement("Tweak");
        tweak.appendChild(document.createTextNode(String.valueOf(tweakValue)));
        layer.appendChild(tweak);

        Element stumps = document.createElement("Stumps");
        layer.appendChild(stumps);

        for (StumpRule sr : committee)
            stumps.appendChild(sr.toXML(document));

        updateFile();
    }

    public static ArrayList<ArrayList<StumpRule>> loadCascadeFromXML(String filePath, ArrayList<Float> tweaks) {
        if (!fileExists(filePath)) {
            System.err.println("Could not load Cascade from file " + filePath + ": file does not exists!");
            System.exit(1);
        }

        ArrayList<ArrayList<StumpRule>> cascade = new ArrayList<>();

        try {
            final DocumentBuilderFactory factory = DocumentBuilderFactory.newInstance();
            final DocumentBuilder builder = factory.newDocumentBuilder();
            final Document document = builder.parse(new File(filePath));
            final Element root = document.getDocumentElement();

            final NodeList layers = root.getChildNodes();

            for (int i = 0; i < layers.getLength(); i++) {
                if (layers.item(i).getNodeType() == Node.ELEMENT_NODE) {
                    ArrayList<StumpRule> committee = new ArrayList<>();
                    final Element layer = (Element) layers.item(i);
                    tweaks.add(Float.valueOf(layer.getElementsByTagName("Tweak").item(0).getTextContent()));
                    final NodeList stumps = layer.getElementsByTagName("Stumps").item(0).getChildNodes();
                    for (int j = 0; j < stumps.getLength(); j++) {
                        if (stumps.item(j).getNodeType() == Node.ELEMENT_NODE) {
                            final Element stump = (Element) stumps.item(j);
                            committee.add(StumpRule.fromXML(stump));
                        }
                    }
                    cascade.add(committee);
                }
            }
        } catch (ParserConfigurationException | SAXException | IOException e) {
            e.printStackTrace();
        }
        return cascade;
    }

    public static void writeCascadeLayer(ArrayList<StumpRule> committee, int round, String fileName) {
        try {
            if (round==0 && fileExists(fileName))
                Utils.deleteFile(fileName);

            PrintWriter writer = new PrintWriter(new FileWriter(fileName, true));

            if (round==0)
                writer.println("double stumps[][4]=");

            for (StumpRule stumpRule : committee) {
                writer.println(stumpRule.featureIndex + ";" + stumpRule.error + ";"
                        + stumpRule.threshold + ";" + stumpRule.toggle);
            }

            writer.flush();
            writer.close();

        } catch (IOException e) {
            System.err.println("Error : Could Not Write committee, aborting");
            e.printStackTrace();
            System.exit(1);
        }
    }

    public static ArrayList<StumpRule> readRule(String fileName) {
        ArrayList<StumpRule> result = new ArrayList<>();

        try {
            BufferedReader br = new BufferedReader(new FileReader(fileName));
            String line = br.readLine();

            while (line != null) {
                line = br.readLine();
                if (line == null || line.equals(""))
                    break;

                String[] parts = line.split(";");

                StumpRule stumpRule = new StumpRule(Long.parseLong(parts[0]),
                        Double.parseDouble(parts[1]),
                        Double.parseDouble(parts[2]),
                        -1,
                        Integer.parseInt(parts[3]));

                result.add(stumpRule);
            }

            br.close();

        } catch (IOException e) {
            System.err.println("Error while reading the list of decisionStumps");
            //e.printStackTrace();
            //System.exit(1);
        }
        return result;
    }

    public static void writeLayerMemory(ArrayList<Integer> layerMemory, ArrayList<Float> tweaks, String fileName) {
        try {

            int layerCount = layerMemory.size();
            PrintWriter writer = new PrintWriter(new FileWriter(fileName, true));

            writer.println(System.lineSeparator());
            writer.println("int layerCount=" + layerCount);
            writer.println("int layerCommitteeSize[]=");

            layerMemory.forEach(writer::println);

            writer.println("float tweaks[]=");

            tweaks.forEach(writer::println);
            /*
            for (int i = 0; i < layerCount; i++) {
                writer.println(tweaks.get(i));
            }
            */

            writer.close();

        } catch (IOException e) {
            System.err.println("Error : Could Not Write layer Memory or tweaks, aborting");
            e.printStackTrace();
            System.exit(1);
        }
    }

    /**
     *
     * @return the layerCount
     * @param fileName the path to the file with the data needed
     * @param layerCommitteeSize ArrayList in which the size of each committe will be stored
     * @param tweaks ArrayList in which the tweaks will be stored
     */
    public static int readLayerMemory(String fileName, ArrayList<Integer> layerCommitteeSize, ArrayList<Float> tweaks) {

        int layercount = 0;

        try {
            BufferedReader br = new BufferedReader(new FileReader(fileName));
            String line = br.readLine();

            while (line != null) {
                line = br.readLine();
                if (line.equals("")) {
                    line = br.readLine();
                    break;
                }
            }

            line = br.readLine();
            String[] parts = line.split("=");
            layercount = Integer.parseInt(parts[1]);
            line = br.readLine();

            for (int i = 0; i < layercount; i++) {
                line = br.readLine();
                layerCommitteeSize.add(Integer.parseInt(line));
            }

            line = br.readLine();

            for (int i = 0; i < layercount; i++) {
                line = br.readLine();
                tweaks.add(Float.parseFloat(line));
            }

            br.close();

        } catch (IOException e) {
            System.err.println("Error while reading the layer memory");
            //e.printStackTrace();
            //System.exit(1);
        }

        return layercount;
    }


    public static ArrayList<StumpRule>[] readLayerMemory(String fileName, ArrayList<Float> tweaks, int[] layerCount) {
        ArrayList<Integer> layerCommitteeSize = new ArrayList<>();
        ArrayList<StumpRule> rules = CascadeSerializer.readRule(fileName);
        layerCount[0] = readLayerMemory(fileName, layerCommitteeSize, tweaks);
        ArrayList<StumpRule>[] cascade = new ArrayList[layerCount[0]];
        int committeeStart = 0;
        for (int i = 0; i < layerCount[0]; i++) {
            cascade[i] = new ArrayList<>();
            for (int committeeIndex = committeeStart; committeeIndex < layerCommitteeSize.get(i) + committeeStart; committeeIndex++) {
                cascade[i].add(rules.get(committeeIndex));
            }
            committeeStart += layerCommitteeSize.get(i);
        }
        return cascade;
    }
}
