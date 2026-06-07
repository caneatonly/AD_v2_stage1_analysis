import java.io.File;
import java.io.FileOutputStream;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.lang.reflect.Array;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.Date;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeSet;

import star.common.Simulation;
import star.common.StarMacro;

/**
 * Exports an AI-readable CFD setup inventory from the currently opened STAR-CCM+ .sim file.
 *
 * Purpose:
 *   - turn a proprietary STAR-CCM+ .sim file into text/CSV artifacts that can be
 *     reviewed outside STAR-CCM+;
 *   - help an engineer or AI assistant inspect the CFD setup, compare cases, and
 *     guide future simulation work;
 *   - support specific downstream tasks such as mesh-independence checks,
 *     convergence audits, report-definition checks, and manuscript evidence tracking.
 *
 * The macro uses reflection so that it remains useful across STAR-CCM+ versions.
 * It writes:
 *   - settings_report.txt: human-readable object tree and properties
 *   - settings_index.csv: searchable path/property/value table
 *   - export_warnings.txt: getters/managers that could not be read
 */
public class ExportSimCfdSettings extends StarMacro {

    private PrintWriter report;
    private PrintWriter csv;
    private PrintWriter warnings;
    private final Map<Object, Boolean> visited = new IdentityHashMap<Object, Boolean>();
    private int csvRows = 0;

    private static final int MAX_OBJECT_DEPTH = 4;
    private static final int MAX_COLLECTION_ITEMS = 80;
    private static final int MAX_VALUE_LENGTH = 600;

    @Override
    public void execute() {
        Simulation sim = getActiveSimulation();
        String stamp = new SimpleDateFormat("yyyyMMdd_HHmmss").format(new Date());
        String simName = safeFileName(readName(sim));
        File outDir = new File(resolvePath("sim_settings_export_" + stamp + "_" + simName));

        if (!outDir.exists()) {
            outDir.mkdirs();
        }

        try {
            report = writer(new File(outDir, "settings_report.txt"));
            csv = writer(new File(outDir, "settings_index.csv"));
            warnings = writer(new File(outDir, "export_warnings.txt"));

            csv.println("section,path,object_class,property,value");

            line("STAR-CCM+ simulation CFD settings export for AI/readable review");
            line("Generated: " + new Date().toString());
            line("Simulation: " + readName(sim));
            line("Output directory: " + outDir.getAbsolutePath());
            line("");

            writeSimulationHeader(sim);
            dumpDiscoveredManagers(sim);
            dumpKnownClassManagers(sim);

            line("");
            line("Export finished. CSV rows: " + csvRows);
            sim.println("CFD settings export written to: " + outDir.getAbsolutePath());
        } catch (Exception ex) {
            sim.println("ExportSimCfdSettings failed: " + ex.getMessage());
            ex.printStackTrace();
        } finally {
            closeQuietly(report);
            closeQuietly(csv);
            closeQuietly(warnings);
        }
    }

    private void writeSimulationHeader(Simulation sim) {
        section("SIMULATION");
        writeCsv("SIMULATION", "/Simulation", sim.getClass().getName(), "presentationName", readName(sim));
        dumpSimpleProperties("SIMULATION", "/Simulation", sim, 0);
    }

    private void dumpDiscoveredManagers(Simulation sim) {
        section("DISCOVERED MANAGERS");
        TreeSet<String> methodNames = new TreeSet<String>();
        for (Method m : sim.getClass().getMethods()) {
            if (isPublicNoArg(m) && m.getName().startsWith("get") && m.getName().endsWith("Manager")) {
                methodNames.add(m.getName());
            }
        }

        for (String methodName : methodNames) {
            try {
                Method m = sim.getClass().getMethod(methodName);
                Object manager = m.invoke(sim);
                if (manager != null) {
                    dumpManager(methodName, "/" + methodName, manager, 0);
                }
            } catch (Exception ex) {
                warn("Could not read manager via " + methodName + ": " + ex.getMessage());
            }
        }
    }

    private void dumpKnownClassManagers(Simulation sim) {
        section("KNOWN CLASS MANAGERS");
        String[] managerClassNames = new String[] {
            "star.meshing.MeshOperationManager",
            "star.common.ContinuumManager",
            "star.common.RegionManager",
            "star.common.InterfaceManager",
            "star.common.ReportManager",
            "star.common.MonitorManager",
            "star.common.SolverManager",
            "star.common.StoppingCriterionManager",
            "star.common.FieldFunctionManager",
            "star.common.CoordinateSystemManager",
            "star.common.SceneManager",
            "star.common.PlotManager",
            "star.common.UnitsManager",
            "star.common.GlobalParameterManager"
        };

        for (String className : managerClassNames) {
            try {
                Class<?> cls = Class.forName(className);
                Method getMethod = findSimulationGetMethod(sim);
                if (getMethod == null) {
                    warn("Simulation.get(Class) was not found; cannot query " + className);
                    return;
                }
                Object manager = getMethod.invoke(sim, cls);
                if (manager != null) {
                    dumpManager(className, "/" + className, manager, 0);
                }
            } catch (ClassNotFoundException ex) {
                warn("Manager class not available in this STAR-CCM+ installation: " + className);
            } catch (Exception ex) {
                warn("Could not query manager class " + className + ": " + ex.getMessage());
            }
        }
    }

    private Method findSimulationGetMethod(Simulation sim) {
        for (Method m : sim.getClass().getMethods()) {
            if (m.getName().equals("get") && m.getParameterTypes().length == 1
                    && m.getParameterTypes()[0].equals(Class.class)) {
                return m;
            }
        }
        return null;
    }

    private void dumpManager(String label, String path, Object manager, int depth) {
        if (manager == null || depth > MAX_OBJECT_DEPTH) {
            return;
        }

        subsection(label + " [" + manager.getClass().getName() + "]");
        writeCsv(label, path, manager.getClass().getName(), "_managerName", readName(manager));
        if (isUnsafeForGetterReflection(label, path, manager)) {
            writeCsv(label, path, manager.getClass().getName(), "_propertiesSkipped",
                    "Getter reflection skipped for safety");
        } else {
            dumpSimpleProperties(label, path, manager, depth);
        }

        Collection<?> objects = readObjects(manager);
        if (objects == null) {
            writeCsv(label, path, manager.getClass().getName(), "_objects", "not available");
            return;
        }

        writeCsv(label, path, manager.getClass().getName(), "_objectCount", Integer.toString(objects.size()));
        int index = 0;
        for (Object obj : objects) {
            if (obj == null) {
                continue;
            }
            if (index >= MAX_COLLECTION_ITEMS) {
                writeCsv(label, path, manager.getClass().getName(), "_truncated",
                        "Only first " + MAX_COLLECTION_ITEMS + " objects exported");
                break;
            }
            String objPath = path + "/" + safePath(readName(obj), index);
            if (isUnsafeForGetterReflection(label, objPath, obj)) {
                dumpObjectHeaderOnly(label, objPath, obj, depth + 1);
            } else {
                dumpObject(label, objPath, obj, depth + 1);
            }
            index++;
        }
    }

    private void dumpObject(String section, String path, Object obj, int depth) {
        if (obj == null || depth > MAX_OBJECT_DEPTH) {
            return;
        }
        if (visited.containsKey(obj)) {
            writeCsv(section, path, obj.getClass().getName(), "_visited", "true");
            return;
        }
        visited.put(obj, Boolean.TRUE);

        line("");
        line(indent(depth) + path);
        line(indent(depth) + "class: " + obj.getClass().getName());
        line(indent(depth) + "name: " + readName(obj));
        writeCsv(section, path, obj.getClass().getName(), "_objectName", readName(obj));

        dumpSimpleProperties(section, path, obj, depth);
        dumpNestedManagers(section, path, obj, depth);
    }

    private void dumpObjectHeaderOnly(String section, String path, Object obj, int depth) {
        if (obj == null || depth > MAX_OBJECT_DEPTH) {
            return;
        }
        line("");
        line(indent(depth) + path);
        line(indent(depth) + "class: " + obj.getClass().getName());
        line(indent(depth) + "name: " + readName(obj));
        line(indent(depth) + "properties: skipped for safety");
        writeCsv(section, path, obj.getClass().getName(), "_objectName", readName(obj));
        writeCsv(section, path, obj.getClass().getName(), "_propertiesSkipped",
                "Getter reflection skipped for safety");
    }

    private void dumpNestedManagers(String section, String path, Object obj, int depth) {
        if (depth >= MAX_OBJECT_DEPTH) {
            return;
        }

        TreeSet<String> names = new TreeSet<String>();
        for (Method m : obj.getClass().getMethods()) {
            if (isPublicNoArg(m) && m.getName().startsWith("get") && m.getName().endsWith("Manager")) {
                names.add(m.getName());
            }
        }

        for (String name : names) {
            try {
                Method m = obj.getClass().getMethod(name);
                Object manager = m.invoke(obj);
                if (manager != null) {
                    dumpManager(section + "." + name, path + "/" + name, manager, depth + 1);
                }
            } catch (Exception ex) {
                warn("Could not read nested manager " + path + "." + name + ": " + ex.getMessage());
            }
        }
    }

    private void dumpSimpleProperties(String section, String path, Object obj, int depth) {
        List<Method> methods = new ArrayList<Method>();
        for (Method m : obj.getClass().getMethods()) {
            if (isReadableGetter(m)) {
                methods.add(m);
            }
        }
        Collections.sort(methods, new Comparator<Method>() {
            public int compare(Method a, Method b) {
                return a.getName().compareTo(b.getName());
            }
        });

        for (Method m : methods) {
            String prop = propertyName(m);
            try {
                Object value = m.invoke(obj);
                if (value == null) {
                    writeCsv(section, path, obj.getClass().getName(), prop, "null");
                    continue;
                }
                if (isSimple(value)) {
                    writeProperty(section, path, obj, depth, prop, valueToString(value));
                } else if (value.getClass().isArray()) {
                    writeProperty(section, path, obj, depth, prop, arrayToString(value));
                } else if (value instanceof Collection<?>) {
                    writeProperty(section, path, obj, depth, prop, collectionToString((Collection<?>) value));
                } else if (looksLikeStarObject(value)) {
                    writeProperty(section, path, obj, depth, prop, readName(value) + " [" + value.getClass().getName() + "]");
                }
            } catch (Exception ex) {
                warn("Could not read property " + path + "." + prop + ": " + ex.getMessage());
            }
        }
    }

    private Collection<?> readObjects(Object manager) {
        try {
            Method getObjects = manager.getClass().getMethod("getObjects");
            Object value = getObjects.invoke(manager);
            if (value instanceof Collection<?>) {
                return (Collection<?>) value;
            }
        } catch (Exception ex) {
            warn("No getObjects() collection for " + manager.getClass().getName() + ": " + ex.getMessage());
        }
        return null;
    }

    private boolean isReadableGetter(Method m) {
        if (!isPublicNoArg(m)) {
            return false;
        }
        String n = m.getName();
        if (!(n.startsWith("get") || n.startsWith("is"))) {
            return false;
        }
        if (n.equals("getClass") || n.equals("getSimulation") || n.equals("getParent")
                || n.equals("getManager") || n.equals("getObjects")) {
            return false;
        }
        if (isDangerousGetterName(n)) {
            return false;
        }
        Class<?> rt = m.getReturnType();
        return rt.isPrimitive()
                || Number.class.isAssignableFrom(rt)
                || Boolean.class.isAssignableFrom(rt)
                || Character.class.isAssignableFrom(rt)
                || String.class.isAssignableFrom(rt)
                || Enum.class.isAssignableFrom(rt)
                || rt.isArray()
                || Collection.class.isAssignableFrom(rt)
                || looksLikeStarClass(rt);
    }

    private boolean isDangerousGetterName(String name) {
        String n = name.toLowerCase();
        return n.contains("value")
                || n.contains("evaluate")
                || n.contains("fieldfunctionvalue")
                || n.contains("child");
    }

    private boolean isUnsafeForGetterReflection(String section, String path, Object obj) {
        String text = ((section == null ? "" : section) + " "
                + (path == null ? "" : path) + " "
                + obj.getClass().getName()).toLowerCase();
        return text.contains("fieldfunction")
                || text.contains("field_function")
                || text.contains("field function");
    }

    private boolean isPublicNoArg(Method m) {
        return Modifier.isPublic(m.getModifiers()) && m.getParameterTypes().length == 0;
    }

    private boolean isSimple(Object value) {
        Class<?> c = value.getClass();
        return c.isPrimitive()
                || value instanceof Number
                || value instanceof Boolean
                || value instanceof Character
                || value instanceof String
                || value instanceof Enum<?>;
    }

    private boolean looksLikeStarObject(Object value) {
        return looksLikeStarClass(value.getClass());
    }

    private boolean looksLikeStarClass(Class<?> cls) {
        Package p = cls.getPackage();
        return p != null && p.getName().startsWith("star.");
    }

    private void writeProperty(String section, String path, Object obj, int depth, String prop, String value) {
        String clipped = clip(value);
        line(indent(depth + 1) + prop + ": " + clipped);
        writeCsv(section, path, obj.getClass().getName(), prop, clipped);
    }

    private void writeCsv(String section, String path, String objectClass, String property, String value) {
        if (csv == null) {
            return;
        }
        csv.println(csv(section) + "," + csv(path) + "," + csv(objectClass) + "," + csv(property) + "," + csv(value));
        csvRows++;
    }

    private String propertyName(Method m) {
        String n = m.getName();
        if (n.startsWith("get") && n.length() > 3) {
            return Character.toLowerCase(n.charAt(3)) + n.substring(4);
        }
        if (n.startsWith("is") && n.length() > 2) {
            return Character.toLowerCase(n.charAt(2)) + n.substring(3);
        }
        return n;
    }

    private String readName(Object obj) {
        String[] methods = new String[] {"getPresentationName", "getName"};
        for (String name : methods) {
            try {
                Method m = obj.getClass().getMethod(name);
                Object value = m.invoke(obj);
                if (value != null) {
                    return value.toString();
                }
            } catch (Exception ignored) {
            }
        }
        return obj.toString();
    }

    private String collectionToString(Collection<?> values) {
        StringBuilder sb = new StringBuilder();
        sb.append("size=").append(values.size()).append(" [");
        int i = 0;
        for (Object v : values) {
            if (i > 0) {
                sb.append("; ");
            }
            if (i >= 20) {
                sb.append("...");
                break;
            }
            sb.append(v == null ? "null" : valueToString(v));
            i++;
        }
        sb.append("]");
        return sb.toString();
    }

    private String arrayToString(Object array) {
        int n = Array.getLength(array);
        StringBuilder sb = new StringBuilder();
        sb.append("length=").append(n).append(" [");
        for (int i = 0; i < Math.min(n, 20); i++) {
            if (i > 0) {
                sb.append("; ");
            }
            Object v = Array.get(array, i);
            sb.append(v == null ? "null" : valueToString(v));
        }
        if (n > 20) {
            sb.append("; ...");
        }
        sb.append("]");
        return sb.toString();
    }

    private String valueToString(Object value) {
        if (value == null) {
            return "null";
        }
        if (isSimple(value)) {
            return value.toString();
        }
        if (looksLikeStarObject(value)) {
            return readName(value) + " [" + value.getClass().getName() + "]";
        }
        return value.toString();
    }

    private String csv(String s) {
        String v = s == null ? "" : s;
        v = v.replace("\"", "\"\"");
        return "\"" + v + "\"";
    }

    private String clip(String value) {
        if (value == null) {
            return "";
        }
        String v = value.replace("\r", " ").replace("\n", " ").trim();
        if (v.length() > MAX_VALUE_LENGTH) {
            return v.substring(0, MAX_VALUE_LENGTH) + "...";
        }
        return v;
    }

    private String safePath(String name, int index) {
        String n = name == null ? "object" : name;
        n = n.replace("/", "_").replace("\\", "_").trim();
        if (n.length() == 0) {
            n = "object";
        }
        return String.format("%03d_%s", index, n);
    }

    private String safeFileName(String name) {
        String n = name == null ? "simulation" : name;
        n = n.replaceAll("[^A-Za-z0-9._-]+", "_");
        if (n.length() == 0) {
            n = "simulation";
        }
        return n;
    }

    private String indent(int depth) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < depth; i++) {
            sb.append("  ");
        }
        return sb.toString();
    }

    private void section(String title) {
        line("");
        line("============================================================");
        line(title);
        line("============================================================");
    }

    private void subsection(String title) {
        line("");
        line("------------------------------------------------------------");
        line(title);
        line("------------------------------------------------------------");
    }

    private void line(String text) {
        if (report != null) {
            report.println(text);
        }
    }

    private void warn(String text) {
        if (warnings != null) {
            warnings.println(text);
        }
    }

    private PrintWriter writer(File file) throws Exception {
        return new PrintWriter(new OutputStreamWriter(new FileOutputStream(file), "UTF-8"));
    }

    private void closeQuietly(PrintWriter w) {
        if (w != null) {
            w.flush();
            w.close();
        }
    }
}
