package  normalization;
import com.sun.xml.internal.bind.v2.schemagen.xmlschema.Union;
import concepts.AtomicConcept;
import concepts.TopConcept;
import connectives.*;
import formula.Formula;

import individual.Individual;
import javafx.util.Pair;
import normalization.Normalize;
import org.semanticweb.HermiT.ReasonerFactory;
import org.semanticweb.owlapi.apibinding.OWLManager;
import org.semanticweb.owlapi.model.*;
import convertion.BackConverter;
import convertion.Converter;
import org.semanticweb.owlapi.model.parameters.Imports;
import org.semanticweb.owlapi.reasoner.InferenceType;
import org.semanticweb.owlapi.reasoner.OWLReasoner;
import org.semanticweb.owlapi.search.EntitySearcher;
import org.semanticweb.owlapi.util.*;
import roles.AtomicRole;
import semantics.MapFunction;
import uk.ac.manchester.cs.owl.owlapi.OWLClassImpl;
import org.semanticweb.HermiT.Reasoner;
import java.io.*;
import java.util.*;
import java.util.stream.Collectors;



public class normalizeALC {
    public static String dirPath = "D:\\Tasks\\DF-ALC\\ontologies\\";
    public static String targetPath = "training\\input\\";
    public static String annotationPath;
    public static String axiomPath;
    public static String aboxPath;
    public static Map<AtomicRole, List<Individual>> r2i_object;
    public static OWLOntologyManager manager;
    public static OWLDataFactory factory;
    public static List<Formula> exists_add(Formula f, Map<AtomicConcept, Set<Individual>> c2i){
        AtomicRole r = (AtomicRole) f.getSubFormulas().get(0);
        AtomicConcept c = (AtomicConcept) f.getSubFormulas().get(1);
        int id = 0;
        if(r2i_object.containsKey(r)) id = r2i_object.get(r).size()+1;

        Individual introduced_i = new Individual(r.getText()+"_individual_"+(id));
        r2i_object.putIfAbsent(r,new ArrayList<>());
        r2i_object.get(r).add(introduced_i);
        Individual negation_i = new Individual(c.getText()+"_individual_negation");
        r2i_object.get(r).add(negation_i);
        List<Formula> formula_list = new ArrayList<>();
        formula_list.add(new RoleAssertion(r, introduced_i, newShuffledSet(c2i.get(c)).iterator().next()));
        formula_list.add(new RoleAssertion(r, negation_i, newShuffledSet(c2i.get(c)).iterator().next()));
        return  formula_list;
    }
    public static List<Formula> forall_add(Formula f, Map<AtomicConcept, Set<Individual>> c2i){
        AtomicRole r = (AtomicRole) f.getSubFormulas().get(0);
        AtomicConcept c = (AtomicConcept) f.getSubFormulas().get(1);
        int id = 0;
        if(r2i_object.containsKey(r)) id = r2i_object.get(r).size()+1;

        Individual introduced_i = new Individual(r.getText()+"_individual_"+(id));
        r2i_object.putIfAbsent(r,new ArrayList<>());
        r2i_object.get(r).add(introduced_i);
        return  Collections.singletonList(new RoleAssertion(r, introduced_i, newShuffledSet(c2i.get(c)).iterator().next()));
    }
    public static <T> Set<T> newShuffledSet(Collection<T> collection) {
        List<T> shuffleMe = new ArrayList<T>(collection);
        Collections.shuffle(shuffleMe);
        return new LinkedHashSet<T>(shuffleMe);
    }

    public static <T> List<T> newShuffledList(Collection<T> collection) {
        List<T> shuffleMe = new ArrayList<T>(collection);
        Collections.shuffle(shuffleMe);
        return new LinkedList<T>(shuffleMe);
    }

    private static void save_abox(List<Formula> abox) throws IOException{
        ArrayList<String> abox_axioms = new ArrayList<>();
        for(Formula f : abox){
//            System.out.println(f);
            if(f==null) continue;
            Set<AtomicConcept> c = f.get_c_sig();
            abox_axioms.add(f.toString());
        }

        save(abox_axioms, aboxPath);

    }

    private  static void save_tbox(OWLOntology o, String ontoName) throws IOException{
        System.out.println("Processing " + ontoName);
        System.out.println("Find " + o.getTBoxAxioms(Imports.EXCLUDED).size() + " TBox axioms");

        HashSet<String> axioms = new HashSet<String>();
        for (OWLAxiom axiom : o.getTBoxAxioms(Imports.EXCLUDED)) {
            if (axiom.getAxiomType() == AxiomType.SUBCLASS_OF) {
                axioms.add(axiom.toString());
            } else if (axiom.getAxiomType() == AxiomType.EQUIVALENT_CLASSES) {
                for (OWLSubClassOfAxiom subaxiom : ((OWLEquivalentClassesAxiom) axiom).asOWLSubClassOfAxioms()) {
                    axioms.add(subaxiom.toString());
                }
            } else if (axiom.getAxiomType() == AxiomType.DISJOINT_CLASSES) {
                for (OWLSubClassOfAxiom subaxiom : ((OWLDisjointClassesAxiom) axiom).asOWLSubClassOfAxioms()) {
                    axioms.add(subaxiom.toString());
                }
            } else if (axiom.getAxiomType() == AxiomType.OBJECT_PROPERTY_RANGE) {
                // Uncomment this to include range
                // axioms.add(((OWLObjectPropertyRangeAxiom) axiom).asOWLSubClassOfAxiom().toString());
            } else if (axiom.getAxiomType() == AxiomType.OBJECT_PROPERTY_DOMAIN) {
                 axioms.add(((OWLObjectPropertyDomainAxiom) axiom).asOWLSubClassOfAxiom().toString());
            } else {
                System.out.println("Unused TBox axiom:");
                System.out.println("  " + axiom);
            }
        }
        System.out.println("Generate " + axioms.size() + " subclassof axioms");
        FileWriter outputFile = new FileWriter(targetPath + ontoName + "_subclassaxioms.txt");
        for (String s: axioms) {
            outputFile.write(s + '\n');
        }
        outputFile.close();
    }

    private static void extract_annotations(OWLOntology ont) throws IOException {
        ArrayList<String> ann_axioms = new ArrayList<>();

        for (OWLEntity e : ont.getSignature()) {
            OWLAnnotation[] annos = EntitySearcher.getAnnotations(e, ont).toArray(OWLAnnotation[]::new);
            for (OWLAnnotation a : annos) {
                OWLAnnotationProperty prop = a.getProperty();
                OWLAnnotationValue val = a.getValue();
                if (val instanceof OWLLiteral) {
                    OWLLiteral lit = (OWLLiteral) val;
                    if ((lit.hasLang() && lit.hasLang("en")) || !lit.hasLang()) {
                        String property_str = prop.getIRI().toString();
                        String e_str = e.getIRI().toString();
                        ann_axioms.add((e_str.replaceAll("\n", " ") + " " +
                                property_str.replaceAll("\n", " ") + " " +
                                ((OWLLiteral) val).getLiteral()).replaceAll("\n", " "));
                    }
                }
            }
        }
        save(ann_axioms, annotationPath);
    }

    private static void save(ArrayList<String> sentences, String path) throws IOException {
        FileWriter aw = new FileWriter(path);
        BufferedWriter bw = new BufferedWriter(aw);
        PrintWriter out = new PrintWriter(bw);
        for (String s : sentences) {
            out.println(s);
        }
        bw.close();
        aw.close();
    }


    public static void save_normalization(File ontology_file, List<Formula> norm_list) throws  Exception{
        FileWriter fw = new FileWriter(new File(targetPath+ontology_file.getName()+"_normalization.txt"));
        System.out.println("----------------------------");
        List<Formula> right_sub; List<Formula> left_sub;
        Set<String> all_formula = new HashSet<>();
        for(Formula formula: norm_list){
            List<Formula> sub_ls = formula.getSubFormulas();
            Formula left_hand = sub_ls.get(0); Formula right_hand = sub_ls.get(1);
            String line = "";
            if(left_hand.getSubFormulas()==null){
                if(right_hand.getSubFormulas()==null){
                    line = line.concat("1|"+left_hand.getName()+"\t"+right_hand.getName()+"|");
                    if(left_hand.getText().contains("\u00AC")) line = line.concat("1 "); else line = line.concat("0 ");
                    if(right_hand.getText().contains("\u00AC")) line = line.concat("1 "); else line = line.concat("0 ");
                }else{
                    right_sub = right_hand.getSubFormulas();
                    if(right_hand instanceof Or){
                        line = line.concat("3|"+left_hand.getName()+"\t"+right_sub.get(0).getName()+"\t"+right_sub.get(1).getName()+"|");
                        if(left_hand.getText().contains("\u00AC")) line = line.concat("1 "); else line = line.concat("0 ");
                        if(right_sub.get(0).getText().contains("\u00AC")) line = line.concat("1 "); else line = line.concat("0 ");
                        if(right_sub.get(1).getText().contains("\u00AC")) line = line.concat("1 "); else line = line.concat("0 ");
                    }else{
                        if(right_hand instanceof Exists){
                            line = line.concat("4|"+left_hand.getName()+"\t"+right_sub.get(0).getName()+"\t"+right_sub.get(1).getName()+"|");
                        }else{
                            line = line.concat("5|"+left_hand.getName()+"\t"+right_sub.get(0).getName()+"\t"+right_sub.get(1).getName()+"|");
                        }
                        if(left_hand.getText().contains("\u00AC")) line = line.concat("1 "); else line = line.concat("0 ");
                        if(right_sub.get(1).getText().contains("\u00AC")) line = line.concat("1 "); else line = line.concat("0 ");
                    }
                }
            }else{
                left_sub = left_hand.getSubFormulas();
                if(left_hand instanceof And){
                    line = line.concat("2|"+left_sub.get(0).getName()+"\t"+left_sub.get(1).getName()+"\t"+right_hand.getName()+"|");
                    if(left_sub.get(0).getText().contains("\u00AC")) line = line.concat("1 "); else line = line.concat("0 ");
                    if(left_sub.get(1).getText().contains("\u00AC")) line = line.concat("1 "); else line = line.concat("0 ");
                    if(right_hand.getText().contains("\u00AC")) line = line.concat("1 "); else line = line.concat("0 ");
                }else{
                    if(left_hand instanceof  Exists){
                        line = line.concat("6|"+left_sub.get(0).getName()+"\t"+left_sub.get(1).getName()+"\t"+right_hand.getName()+"|");
                    }else{
                        line = line.concat("7|"+left_sub.get(0).getName()+"\t"+left_sub.get(1).getName()+"\t"+right_hand.getName()+"|");
                    }
                    if(left_sub.get(1).getText().contains("\u00AC")) line = line.concat("1 "); else line = line.concat("0 ");
                    if(right_hand.getText().contains("\u00AC")) line = line.concat("1 "); else line = line.concat("0 ");
                }
            }
            if(all_formula.contains(line)) continue;
            all_formula.add(line);
            fw.write(line+"\n");
            fw.flush();
        }
        fw.close();
    }

    public static void infer(OWLOntology o) throws OWLOntologyCreationException {
        OWLReasoner reasoner = new ReasonerFactory().createReasoner(o);
        List<InferredAxiomGenerator<? extends OWLAxiom>> generators = new ArrayList<>();
        generators.add(new InferredSubClassAxiomGenerator());
        generators.add(new InferredClassAssertionAxiomGenerator());
        generators.add(new InferredDisjointClassesAxiomGenerator() {
            boolean precomputed = false;

            protected void addAxioms(OWLClass entity, OWLReasoner reasoner, OWLDataFactory dataFactory, Set<OWLDisjointClassesAxiom> result) {
                if (!precomputed) {
                    reasoner.precomputeInferences(InferenceType.DISJOINT_CLASSES);
                    precomputed = true;
                }
                for (OWLClass cls : reasoner.getDisjointClasses(entity).getFlattened()) {
                    result.add(dataFactory.getOWLDisjointClassesAxiom(entity, cls));
                }
            }
        });

        InferredOntologyGenerator iog = new InferredOntologyGenerator(reasoner, generators);

        OWLOntologyManager manager = OWLManager.createOWLOntologyManager();
        OWLOntology inferredAxiomsOntology = manager.createOntology();

        iog.fillOntology(manager.getOWLDataFactory(), inferredAxiomsOntology);
        manager.addAxioms(o, inferredAxiomsOntology.getLogicalAxioms());
        for (OWLNamedIndividual i : inferredAxiomsOntology.getIndividualsInSignature()) {
            manager.addAxioms(o, inferredAxiomsOntology.getClassAssertionAxioms(i));
            manager.addAxioms(o, inferredAxiomsOntology.getObjectPropertyAssertionAxioms(i));
        }

        System.out.println("after inferring axiom amount: " + o.getAxiomCount());
        System.out.println("after inferring concept amount: " + o.getClassesInSignature().size());
        System.out.println("after inferring individual amount: " + o.getIndividualsInSignature().size());
    }

    public static void main(String[] args) throws Exception {
        if(args.length == 2 ){
            dirPath = args[0];
            targetPath = args[1];
        }
        File directory = new File(dirPath);
        boolean inferflag = true;
        ArrayList file_list = new ArrayList<>();
        for(File ontology_file:directory.listFiles()){
            annotationPath = targetPath + ontology_file.getName() + "_annotation.txt";
            axiomPath = targetPath + ontology_file.getName() + "_axiom.txt";
            aboxPath = targetPath + ontology_file.getName() + "_abox.txt";

            try {
                System.out.println(ontology_file.getName());
                manager = OWLManager.createOWLOntologyManager();
                factory = manager.getOWLDataFactory();
                OWLOntology o;
                o = manager.loadOntologyFromOntologyDocument(ontology_file);
                System.out.println("original axiom amount: " + o.getAxiomCount());
                System.out.println("original concept amount: " + o.getClassesInSignature().size());
                System.out.println("original individual amount: " + o.getIndividualsInSignature().size());

                // Save TBox axioms
                save_tbox(o,ontology_file.getName());
                // (Optional) Infer disjoint axioms and assertions.
                if (inferflag) {
                    infer(o);
                }


                Converter ct = new Converter();
                Normalize normalize = new Normalize();

                // Extract the ALC fragment
                List<Formula> o_alc = ct.OntologyConverter(o, "alc");
                System.out.println(o_alc);

                // Normalize the ontology
                List<Formula> alc_norm = normalize.ALCNormalize(o_alc);
                alc_norm.addAll(ct.ABoxConvertor(o));
                System.out.println("alc_norm: " + alc_norm);

                BackConverter bc = new BackConverter();
                OWLOntology alc_normed_o = bc.toOWLOntology(alc_norm);

                // To complete the assertions of introduced definers
                if(inferflag){
                    infer(alc_normed_o);
                }
                alc_normed_o.saveOntology(new FileOutputStream(new File(targetPath+ontology_file.getName()+".inferred.owl")));

                List<Formula> inferred_alc_norm = ct.OntologyConverter(alc_normed_o,"alc");

                // Save ABox
                save_abox(ct.ABoxConvertor((alc_normed_o)));

                Set<String> concept_set = new HashSet<>();
                Set<String> role_set = new HashSet<>();
                for (OWLClass c : alc_normed_o.getClassesInSignature()) {
                    concept_set.add(c.getIRI().toString());
                }
                for (Formula f : alc_norm) {
                    for (AtomicConcept c : f.get_c_sig()) {
                        if (c.getText().contains("\u00AC")) concept_set.add(c.getText().substring(1));
                        else concept_set.add(c.getText());
                    }

                    for (AtomicRole r : f.get_r_sig()) {
                        role_set.add(r.getText());
                    }
                }
                System.out.println("role: " + role_set);
                FileWriter fw = new FileWriter(new File(targetPath + ontology_file.getName() + "_concepts.txt"));
                for (String c : concept_set) {
                    fw.write(c + "\n");
                    fw.flush();
                }
                fw.close();
                fw = new FileWriter(new File(targetPath + ontology_file.getName() + "_roles.txt"));
                for (String r : role_set) {
                    fw.write(r + "\n");
                    fw.flush();
                }
                fw.close();
                fw = new FileWriter(new File(targetPath + ontology_file.getName() + "_individuals.txt"));
                for (OWLNamedIndividual i : alc_normed_o.getIndividualsInSignature()) {
                    String r = i.getIRI().toString();
                    fw.write(r + "\n");
                    fw.flush();
                }
                fw.close();

                System.out.println("After normalization concept amount: " + concept_set.size());
                System.out.println("============================================================================================================================================");
                save_normalization(ontology_file, inferred_alc_norm);

                System.out.println("Finish!");
            }catch(Exception ex){
                ex.printStackTrace(new java.io.PrintStream(System.out));

            }



        }
        System.out.println(file_list);

    }
}
