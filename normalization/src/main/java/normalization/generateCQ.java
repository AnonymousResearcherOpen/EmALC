package normalization;

import org.semanticweb.HermiT.ReasonerFactory;
import org.semanticweb.owlapi.apibinding.OWLManager;
import org.semanticweb.owlapi.model.*;
import org.semanticweb.owlapi.reasoner.InferenceType;
import org.semanticweb.owlapi.reasoner.NodeSet;
import org.semanticweb.owlapi.reasoner.OWLReasoner;
import org.semanticweb.owlapi.util.*;

import java.io.File;
import java.io.FileWriter;
import java.util.*;


public class generateCQ {
    public static String dirPath = "training/ontologies/";
    public static String outPath = "training/input/";
    public static ArrayList<OWLClass> concepts;
    public static  ArrayList<OWLObjectProperty> roles;
    public static OWLNamedIndividual[] individuals;
    public static OWLOntologyManager manager;
    public static OWLDataFactory factory;
    public static OWLClassExpression QueryGenerator(int mode, int depths){
        Random rand = new Random();
        int idx = rand.nextInt(concepts.size());
        OWLClassExpression q = concepts.get(idx);
        Set<Integer> all_c = new HashSet<>();

        int cnt = depths-1;
        if(mode == 1){
            all_c.add(idx);
            while(cnt != 0){
                idx = rand.nextInt(concepts.size());
                if(all_c.contains(idx)) continue;
                q = factory.getOWLObjectIntersectionOf(q,concepts.get(idx));
                all_c.add(idx);
                cnt --;
            }
        }else if(mode == 2){
            q = factory.getOWLObjectSomeValuesFrom(roles.get(rand.nextInt(roles.size())), q);
            while(cnt != 0){
                idx = rand.nextInt(concepts.size());
                if(all_c.contains(idx)) continue;
                q = factory.getOWLObjectIntersectionOf(q,concepts.get(idx));
                all_c.add(idx);
                cnt --;
            }

        }else{
            q = factory.getOWLObjectAllValuesFrom(roles.get(rand.nextInt(roles.size())), q);
            while(cnt != 0){
                idx = rand.nextInt(concepts.size());
                if(all_c.contains(idx)) continue;
                q = factory.getOWLObjectIntersectionOf(q,concepts.get(idx));
                all_c.add(idx);
                cnt --;
            }
        }
        return q;
    }

    public static void infer(OWLOntology o) throws OWLOntologyCreationException {
        OWLReasoner reasoner = new ReasonerFactory().createReasoner(o);
        List<InferredAxiomGenerator<? extends OWLAxiom>> generators = new ArrayList<>();
        generators.add(new InferredSubClassAxiomGenerator());
        generators.add(new InferredClassAssertionAxiomGenerator());
//        generators.add(new);
        generators.add(new InferredDisjointClassesAxiomGenerator() {
            boolean precomputed = false;

            protected void addAxioms(OWLClass entity, OWLReasoner reasoner, OWLDataFactory dataFactory, Set<OWLDisjointClassesAxiom> result) {
                if (!precomputed) {
                    reasoner.precomputeInferences(InferenceType.DISJOINT_CLASSES);
                    precomputed = true;
                }
            }
        });

        InferredOntologyGenerator iog = new InferredOntologyGenerator(reasoner, generators);
        OWLOntologyManager manager = OWLManager.createOWLOntologyManager();
        OWLOntology inferredAxiomsOntology = manager.createOntology();
        iog.fillOntology(manager.getOWLDataFactory(), inferredAxiomsOntology);
        manager.addAxioms(o, inferredAxiomsOntology.getLogicalAxioms());
        for (OWLNamedIndividual i : inferredAxiomsOntology.getIndividualsInSignature()) {
//            System.out.println("inferred class assertion: "+inferredAxiomsOntology.getClassAssertionAxioms(i));
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
            outPath = args[1];
        }
        File directory = new File(dirPath);
        int depths = 3;
        for (File ontology_file:directory.listFiles()) {
            String ontology_name = ontology_file.getName();
            System.out.println(ontology_name);
            manager = OWLManager.createOWLOntologyManager();
            factory = manager.getOWLDataFactory();

            OWLOntology o;

            o = manager.loadOntologyFromOntologyDocument(new File(dirPath+ontology_name));
            OWLReasoner reasoner = new ReasonerFactory().createReasoner(o);
            System.out.println("original axiom amount: " + o.getAxiomCount());
            System.out.println("original concept amount: " + o.getClassesInSignature().size());
            System.out.println("original individual amount: " + o.getIndividualsInSignature().size());
            System.out.println("original role amount: "+o.getObjectPropertiesInSignature().size());
            individuals = o.getIndividualsInSignature().toArray(new OWLNamedIndividual[o.getIndividualsInSignature().size()]);
            infer(o);
            concepts = new ArrayList<>(); roles = new ArrayList<>();
            for(OWLNamedIndividual ind:individuals){
                for(OWLClassAssertionAxiom ca:o.getClassAssertionAxioms(ind)){
                    if(concepts.contains(ca.getClassExpression())) continue;
                    concepts.add((OWLClass)ca.getClassExpression());
                }
                for(OWLObjectPropertyAssertionAxiom opa:o.getObjectPropertyAssertionAxioms(ind)){
                    for(OWLObjectProperty r:opa.getObjectPropertiesInSignature()){
                        if(roles.contains(r)) continue;
                        roles.add((OWLObjectProperty)r);
                    }
                }

            }
            System.out.println("concept size: "+ concepts.size());
            System.out.println("role size: "+roles.size());
            FileWriter fw = new FileWriter(new File(outPath+ontology_name+".depth_"+depths+".queries"));
            FileWriter fw2 = new FileWriter(new File(outPath+ontology_name+".depth_"+depths+".answers"));
            for(int mode=1;mode<=2;mode++){
                if((roles.size()==0) && (mode == 2)) continue;
                ArrayList<OWLClassExpression> queries = new ArrayList<>();
                ArrayList<NodeSet<OWLNamedIndividual>> answers = new ArrayList<>();
                int number = 10;
                while(number!=0){
                    OWLClassExpression query = QueryGenerator(mode,depths);
                    NodeSet<OWLNamedIndividual> answer =  reasoner.getInstances(query);
                    if(!answer.isEmpty()) {
                        number --;
                        queries.add(query);
                        answers.add(answer);
                    }

                }


                System.out.println(queries);
                System.out.println(answers);


                for(OWLClassExpression q:queries){
                    String line = ""; line += mode+"|";
                    if(mode == 1){
                        for(OWLClassExpression c:q.asConjunctSet()){
                            OWLClass cc = (OWLClass) c;
                            line += cc.getIRI().toString()+" ";
                        }
                    }else if(mode==2){
                        OWLClassExpression[] clist = q.asConjunctSet().toArray(new OWLClassExpression[q.asConjunctSet().size()]);
                        for(int i=0;i<clist.length-1;i++){
                            OWLClass c = (OWLClass) clist[i];
                            line += c.getIRI().toString()+" ";
                        }
                        line += clist[clist.length-1].getObjectPropertiesInSignature().iterator().next().getIRI().toString()+" ";
                        line += clist[clist.length-1].getClassesInSignature().iterator().next().getIRI().toString()+" ";
                    }else{
                        OWLClassExpression[] clist = q.asConjunctSet().toArray(new OWLClassExpression[q.asConjunctSet().size()]);
                        for(int i=0;i<clist.length-1;i++){
                            OWLClass c = (OWLClass) clist[i];
                            line += c.getIRI().toString()+" ";
                        }
                        line += clist[0].getObjectPropertiesInSignature().iterator().next().getIRI().toString()+" ";
                        line += clist[0].getClassesInSignature().iterator().next().getIRI().toString()+" ";
                    }
                    line += "\n";
                    fw.write(line);
                }
                fw.flush();



                for(NodeSet<OWLNamedIndividual> nodeset:answers){

                    String line = ""; line += mode + "|";
                    int size = (int) nodeset.entities().count();
                    OWLNamedIndividual[] is =  nodeset.getFlattened().toArray(new OWLNamedIndividual[size]);
                    for(OWLNamedIndividual i:is){
                        line += i.getIRI().toString() + " ";
                    }
                    fw2.write(line+"\n");

                }
                fw2.flush();
            }
            fw.close();fw2.close();
        }
    }
}
