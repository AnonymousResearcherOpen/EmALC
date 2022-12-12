/*
1 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package convertion;

import concepts.AtomicConcept;
import concepts.TopConcept;
import connectives.*;
import data.DataProperty;
import data.Literal;
import formula.Formula;
import individual.Individual;
import javafx.beans.property.ObjectProperty;
import org.semanticweb.owlapi.model.*;
import roles.AtomicRole;
import roles.RoleExpression;
import roles.TopRole;

import java.net.DatagramPacket;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.*;

import org.semanticweb.owlapi.apibinding.OWLManager;
import uk.ac.manchester.cs.owl.owlapi.*;



public class Converter {
	public Map<String, Formula> objectMap;
	public Converter() {
		objectMap = new HashMap<>();
	}

	public AtomicConcept getConceptfromClass(OWLEntity owlClass) {
		return new AtomicConcept(owlClass.getIRI().toString());
	}
	
	public AtomicRole getRoleFromObjectProperty(OWLEntity owlObjectProperty) {		
		return new AtomicRole(owlObjectProperty.getIRI().toString());
	}
	
	public Set<AtomicConcept> getConceptsfromClasses(Set<OWLClass> class_set) {

		Set<AtomicConcept> concept_set = new LinkedHashSet<>();

		for (OWLClass owlClass : class_set) {
			concept_set.add(getConceptfromClass(owlClass));
		}

		return concept_set;
	}

	public Set<AtomicRole> getRolesfromObjectProperties(Set<OWLObjectProperty> op_set) {

		Set<AtomicRole> role_set = new LinkedHashSet<>();

		for (OWLObjectProperty owlObjectProperty : op_set) {
			role_set.add(getRoleFromObjectProperty(owlObjectProperty));
		}

		return role_set;
	}
				
	public List<AtomicConcept> getConceptsInSignature(OWLOntology ontology) {

		List<AtomicConcept> concept_list = new ArrayList<>();
		Set<OWLClass> class_set = ontology.getClassesInSignature();

		for (OWLClass owlClass : class_set) {
			concept_list.add(getConceptfromClass(owlClass));
		}

		return concept_list;
	}
		
	public List<AtomicRole> getRolesInSignature(OWLOntology ontology) {

		List<AtomicRole> role_list = new ArrayList<>();
		Set<OWLObjectProperty> op_set = ontology.getObjectPropertiesInSignature();

		for (OWLObjectProperty owlObjectProperty : op_set) {
			role_list.add(getRoleFromObjectProperty(owlObjectProperty));
		}
		return role_list;
	}
	public List<Formula> RightSubformulasConverter(List<Formula> formulaList){
		List<Formula> ans = new ArrayList<>();
		for(Formula formula : formulaList){
			ans.add(RightSubformulaConverter(formula));
		}
		return ans;
	}
	public Formula RightSubformulaConverter(Formula formula){
		if(formula instanceof AtomicConcept || formula instanceof AtomicRole){
			return formula;
		}
		else if(formula instanceof Inclusion){
			return new Inclusion(RightSubformulaConverter(formula.getSubFormulas().get(0)),
					RightSubformulaConverter(formula.getSubFormulas().get(1)));
		}
		else if(formula instanceof Exists){
			return new Exists(formula.getSubFormulas().get(0),RightSubformulaConverter(formula.getSubFormulas().get(1)));
		}
		else if(formula instanceof And){
			Set<Formula> and = new LinkedHashSet<>();
			for(Formula f: formula.getSubformulae()){
				if(f instanceof And){
					for(Formula i:f.getSubformulae()){
						and.add(RightSubformulaConverter(i));
					}
				}
				else{
					and.add(RightSubformulaConverter(f));
				}
			}
			return  And.getAnd(and);
		}
		else{
			return formula;
		}
	}
	public List<Formula> OntologyConverter(OWLOntology ontology, String expressivity) {

		List<Formula> formula_list = new ArrayList<>();		
		Set<OWLLogicalAxiom> owlAxiom_set = ontology.getLogicalAxioms();
		
		long startTime1 = System.currentTimeMillis();
		
		for (OWLAxiom owlAxiom : owlAxiom_set) {
			List<Formula> temp_list = AxiomConverter(owlAxiom, expressivity);
			temp_list = RightSubformulasConverter(temp_list);
			for (Formula formula : temp_list) {
				Formula subsume = formula.getSubFormulas().get(0);
				Formula subsumer = formula.getSubFormulas().get(1);

				if (subsumer instanceof And ){
					if(owlAxiom instanceof OWLEquivalentClassesAxiom){
						formula_list.add(new Inclusion(subsumer, subsume));
					}
					Set<Formula> conjunct_set = subsumer.getSubformulae();

					for (Formula conjunct : conjunct_set) {
						//if(conjunct == null) {System.out.println("iiiiii");System.out.println(subsumer);}
						Formula inclusion = new Inclusion(subsume, conjunct);
						formula_list.add(inclusion);
					}

				}else if(subsume instanceof Or){
					if(owlAxiom instanceof OWLEquivalentClassesAxiom){
						formula_list.add(new Inclusion(subsumer, subsume));
					}
					Set<Formula> disjunct_set = subsume.getSubformulae();
					for(Formula disjunct: disjunct_set){
						Formula inclusion = new Inclusion(disjunct, subsumer);
						formula_list.add(inclusion);
					}
				} else {
					formula_list.add(formula);
				}


			}
			//formula_list.addAll(temp_list);
		}
		
		long endTime1 = System.currentTimeMillis();
		
		System.out.println("Convertion Duration = " + (endTime1 - startTime1) + " millis");
		return formula_list;
	}
	
	public List<Formula> AxiomsConverter(Set<OWLLogicalAxiom> owlAxiom_set, String expressivity) {

		List<Formula> formula_list = new ArrayList<>();
		
		long startTime1 = System.currentTimeMillis();

		for (OWLAxiom owlAxiom : owlAxiom_set) {
			List<Formula> temp_list = AxiomConverter(owlAxiom,expressivity);
			temp_list = RightSubformulasConverter(temp_list);
			for (Formula formula : temp_list) {
				Formula subsume = formula.getSubFormulas().get(0);
				Formula subsumer = formula.getSubFormulas().get(1);
				if (subsumer instanceof And) {
					//System.out.println("formula = " + formula);
					Formula subsumee = formula.getSubFormulas().get(0);
					Set<Formula> conjunct_set = subsumer.getSubformulae();
					for (Formula conjunct : conjunct_set) {
						if(conjunct == null) System.out.println("jjjjj");

						Formula inclusion = new Inclusion(subsumee, conjunct);
						formula_list.add(inclusion);
					}
					
				} else if(subsume instanceof Or){
					Formula subsumerr = formula.getSubFormulas().get(1);
					Set<Formula> disjunct_set = subsume.getSubformulae();
					for(Formula disjunct: disjunct_set){
						Formula inclusion = new Inclusion(disjunct, subsumer);
						formula_list.add(inclusion);
					}
				}else {
					formula_list.add(formula);
				}
			}
			//formula_list.addAll(temp_list);
		}
		
		long endTime1 = System.currentTimeMillis();
		
		System.out.println("Convertion Duration = " + (endTime1 - startTime1) + " millis");

		return formula_list;
	}
	
	
	/*public List<Formula> AxiomsConverter(Set<OWLAxiom> owlAxiom_set) {

		List<Formula> formula_list = new ArrayList<>();

		for (OWLAxiom owlAxiom : owlAxiom_set) {
			formula_list.addAll(AxiomConverter(owlAxiom));
		}

		return formula_list;
	}*/
	
		
	private List<Formula> AxiomConverter(OWLAxiom axiom, String expressivity) {
		if (axiom instanceof OWLSubClassOfAxiom) {
			OWLSubClassOfAxiom owlSCOA = (OWLSubClassOfAxiom) axiom;
			Formula l; Formula r;
			if(expressivity.equals("el")){
				l = ClassExpressionConverter(owlSCOA.getSubClass());
				r = ClassExpressionConverter(owlSCOA.getSuperClass());
			}else{
				l = ALCClassExpressionConverter(owlSCOA.getSubClass());
				r = ALCClassExpressionConverter(owlSCOA.getSuperClass());
			}

			if(l == null || r == null) return  Collections.emptyList();
			Formula converted = new Inclusion(l, r);
			return Collections.singletonList(converted);

		} else if (axiom instanceof OWLEquivalentClassesAxiom) {
			OWLEquivalentClassesAxiom owlECA = (OWLEquivalentClassesAxiom) axiom;
			System.out.println("OWLEquivalentClassesAxiom: "+owlECA);
//			Collection<OWLSubClassOfAxiom> owlSubClassOfAxioms = owlECA.asOWLSubClassOfAxioms();
//			List<Formula> converted = new ArrayList<>();
//			for (OWLSubClassOfAxiom owlSC : owlSubClassOfAxioms) {
//				List<Formula> temp = AxiomConverter(owlSC, expressivity);
////				System.out.println("temp: "+temp+owlSC);
////				if(temp.size() == 0) return Collections.emptyList();
//				converted.addAll(temp);
//			}
			OWLClassExpression[] class_expressions = new OWLClassExpression[2];
			owlECA.getClassExpressions().toArray(class_expressions);
			Formula l; Formula r;
			if(expressivity.equals("el")){
				l = ClassExpressionConverter(class_expressions[0]);
				r = ClassExpressionConverter(class_expressions[1]);
			}else{
				l = ALCClassExpressionConverter(class_expressions[0]);
				r = ALCClassExpressionConverter(class_expressions[1]);
			}
			Formula converted = new Equivalence(l,r);
			return Collections.singletonList(converted);

		} else if(axiom instanceof  OWLDisjointClassesAxiom){
			OWLDisjointClassesAxiom owlDCA = (OWLDisjointClassesAxiom) axiom;
			Set<OWLClassExpression> cs = owlDCA.getClassExpressionsMinus();
			Iterator iter = cs.iterator();
			Formula l; Formula r;Formula converted;
			if(expressivity.equals("el")){
				l = ClassExpressionConverter((OWLClassExpression)iter.next());
				r = ClassExpressionConverter((OWLClassExpression)iter.next());
				if(l == null || r == null) return  Collections.emptyList();
				converted = new Disjoint(l, r);
			}else{
				l = ALCClassExpressionConverter((OWLClassExpression)iter.next());
				r = ALCClassExpressionConverter((OWLClassExpression)iter.next());
				if(l == null || r == null) return  Collections.emptyList();
				Set<Formula> fs = new HashSet<>();
				fs.add(l);fs.add(r);
				converted = new Inclusion(And.getAnd(fs), new AtomicConcept("http://www.w3.org/2002/07/owl#Nothing"));
				System.out.println("Disjoint: "+owlDCA);
			}


			return Collections.singletonList(converted);
		}
		else if (axiom instanceof OWLObjectPropertyDomainAxiom) {
			OWLObjectPropertyDomainAxiom owlOPDA = (OWLObjectPropertyDomainAxiom) axiom;
			OWLSubClassOfAxiom owlSCOA = owlOPDA.asOWLSubClassOfAxiom();
			System.out.println(owlSCOA);
			return AxiomConverter(owlSCOA, expressivity);

		}
		// As the conversion of OWLObjectPropertyRangeAxiom is not good here, we ignored the conversion
//		else if (axiom instanceof OWLObjectPropertyRangeAxiom) {
//			OWLObjectPropertyRangeAxiom owlOPRA = (OWLObjectPropertyRangeAxiom) axiom;
//			OWLSubClassOfAxiom owlSCOA = owlOPRA.asOWLSubClassOfAxiom();
//			System.out.println(owlSCOA);
//			return AxiomConverter(owlSCOA, expressivity);
//
//		}
//		else if (axiom instanceof OWLSubObjectPropertyOfAxiom) {
//			OWLSubObjectPropertyOfAxiom owlSOPOA = (OWLSubObjectPropertyOfAxiom) axiom;
//			Formula l = RoleExpressionConverter(owlSOPOA.getSubProperty());
//			Formula r = RoleExpressionConverter(owlSOPOA.getSuperProperty());
//			if(l == null || r == null) return Collections.emptyList();
//			Formula converted = new Inclusion(l, r);
//			return Collections.singletonList(converted);
//
//		} else if (axiom instanceof OWLEquivalentObjectPropertiesAxiom) {
//			OWLEquivalentObjectPropertiesAxiom owlEOPA = (OWLEquivalentObjectPropertiesAxiom) axiom;
//			Collection<OWLSubObjectPropertyOfAxiom> owlSOPOAs = owlEOPA.asSubObjectPropertyOfAxioms();
//			List<Formula> converted = new ArrayList<>();
//			for (OWLSubObjectPropertyOfAxiom owlSOPOA : owlSOPOAs) {
//				converted.addAll(AxiomConverter(owlSOPOA));
//			}
//
//			return converted;
//
//		}

		return Collections.emptyList();
	}


	public Formula ClassExpressionConverter(OWLClassExpression concept) {
	
		if (concept.isTopEntity()) {
//			return TopConcept.getInstance();
			return new AtomicConcept("http://www.w3.org/2002/07/owl#Thing");
		} else if (concept instanceof OWLClass) {
			OWLClass owlClass = (OWLClass) concept;
			return new AtomicConcept(owlClass.getIRI().toString());
//			if(objectMap.containsKey(owlClass.getIRI().toString())==false)
//				objectMap.put(owlClass.getIRI().toString(),new AtomicConcept(owlClass.getIRI().toString()));
//			return objectMap.get(owlClass.getIRI().toString());
		} else if (concept instanceof OWLObjectSomeValuesFrom) {
			OWLObjectSomeValuesFrom owlOSVF = (OWLObjectSomeValuesFrom) concept;
			RoleExpression r = RoleExpressionConverter(owlOSVF.getProperty());
			Formula c = ClassExpressionConverter(owlOSVF.getFiller());
			if(r == null || c == null) return null;
			return new Exists(r,c);

		} else if (concept instanceof OWLObjectIntersectionOf) {
			OWLObjectIntersectionOf owlOIO = (OWLObjectIntersectionOf) concept;
			Set<Formula> conjunct_set = new LinkedHashSet<>();
			for (OWLClassExpression conjunct : owlOIO.getOperands()) {

				Formula temp = ClassExpressionConverter(conjunct);
				if(temp == null) return null;

				conjunct_set.add(temp);

			}
			return  And.getAnd(conjunct_set);

		}
		else if(concept instanceof OWLObjectHasValue){
			OWLObjectHasValue owlOHV = (OWLObjectHasValue) concept;
			RoleExpression r = RoleExpressionConverter(owlOHV.getProperty());
			Formula c = new AtomicConcept(((OWLClass)owlOHV.getFiller()).getIRI().toString());
			if(r == null || c == null) return null;
			return new Exists(r,c);
		}
		else if(concept instanceof OWLObjectAllValuesFrom){
			OWLObjectAllValuesFrom owlOAV = (OWLObjectAllValuesFrom) concept;
			RoleExpression r = RoleExpressionConverter((owlOAV.getProperty()));
			Formula c = new AtomicConcept(((OWLClass)owlOAV.getFiller()).getIRI().toString());
			if(r == null || c == null) return null;
			return new Forall(r,c);
		}
		else if(concept instanceof OWLDataHasValue){
			OWLDataHasValue owlDHV = (OWLDataHasValue) concept;
			DataProperty r = new DataProperty(((OWLDataProperty)owlDHV.getProperty()).getIRI().toString());
			Literal c = new Literal((owlDHV.getFiller().getLiteral()));
			if(r == null || c == null) return null;
			return new HasValue(r,c);
		}
		return null;
		//return TopConcept.getInstance();
	}

	public Formula ALCClassExpressionConverter(OWLClassExpression concept) {
		System.out.println(concept);
		if (concept.isTopEntity()) {
//			return TopConcept.getInstance();
			return new AtomicConcept("http://www.w3.org/2002/07/owl#Thing");
		}
		else if (concept instanceof OWLClass) {
			OWLClass owlClass = (OWLClass) concept;
			return new AtomicConcept(owlClass.getIRI().toString());
//			if(objectMap.containsKey(owlClass.getIRI().toString())==false)
//				objectMap.put(owlClass.getIRI().toString(),new AtomicConcept(owlClass.getIRI().toString()));
//			return objectMap.get(owlClass.getIRI().toString());
		} else if (concept instanceof OWLObjectSomeValuesFrom) {
			OWLObjectSomeValuesFrom owlOSVF = (OWLObjectSomeValuesFrom) concept;
			RoleExpression r = RoleExpressionConverter(owlOSVF.getProperty());
			Formula c = null;
//			if(owlOSVF.getFiller() instanceof OWLClass)
//				c = new AtomicConcept(((OWLClass)owlOSVF.getFiller()).getIRI().toString());
//			else if(owlOSVF.getFiller() instanceof OWLObjectComplementOfImpl){
//				OWLObjectComplementOfImpl coi = (OWLObjectComplementOfImpl) owlOSVF.getFiller();
//				OWLClassExpression nn = coi.getNNF();
////				System.out.println("coi: "+coi);
////				System.out.println("nn: "+nn);
//				System.out.println("nn: "+nn.getComplementNNF());
//				if(nn.getComplementNNF()  instanceof OWLClass){
//					c = new AtomicConcept("\u00AC"+((OWLClass) nn.getComplementNNF()).getIRI().toString());
//					c.setNeg();
//				}else {
//					return ALCClassExpressionConverter(nn);
//				}
//			}else{
				c = ALCClassExpressionConverter(owlOSVF.getFiller());
//			}
			if(r == null || c == null) return null;
			return new Exists(r,c);

		} else if (concept instanceof OWLObjectIntersectionOf) {
			OWLObjectIntersectionOf owlOIO = (OWLObjectIntersectionOf) concept;
			Set<Formula> conjunct_set = new LinkedHashSet<>();
			for (OWLClassExpression conjunct : owlOIO.getOperands()) {
				Formula temp = ALCClassExpressionConverter(conjunct);
				if(temp == null) return null;
				conjunct_set.add(temp);
			}
			return  And.getAnd(conjunct_set);

		}else if(concept instanceof OWLObjectUnionOf){
			OWLObjectUnionOf owlOUO = (OWLObjectUnionOf) concept;
			Set<Formula> union_set = new LinkedHashSet<>();
			for(OWLClassExpression union:owlOUO.getOperands()){
				Formula temp = ALCClassExpressionConverter(union);
				if(temp == null) return null;
				union_set.add(temp);
			}
			return Or.getOr(union_set);
		}
		else if(concept instanceof OWLObjectHasValue){
			OWLObjectHasValue owlOHV = (OWLObjectHasValue) concept;
			RoleExpression r = RoleExpressionConverter(owlOHV.getProperty());
			Formula c = null;
			if(owlOHV.getFiller() instanceof OWLClass)
				c = new AtomicConcept(((OWLClass)owlOHV.getFiller()).getIRI().toString());
			else if(owlOHV.getFiller() instanceof OWLNamedIndividualImpl)
				c = new AtomicConcept(((OWLNamedIndividualImpl)owlOHV.getFiller()).getIRI().toString());
			else if(owlOHV.getFiller() instanceof OWLObjectComplementOfImpl){
				OWLObjectComplementOfImpl coi = (OWLObjectComplementOfImpl) owlOHV.getFiller();

				OWLClassExpression nn = coi.getNNF();
				if(nn.getComplementNNF()  instanceof OWLClass){
					c = new AtomicConcept("\u00AC"+((OWLClass) nn.getComplementNNF()).getIRI().toString());
					c.setNeg();
				}else {
					return ALCClassExpressionConverter(nn);
				}
			}

			if(r == null || c == null) return null;
			return new Exists(r,c);
		}
		else if(concept instanceof OWLObjectAllValuesFrom){
			OWLObjectAllValuesFrom owlOAV = (OWLObjectAllValuesFrom) concept;
			RoleExpression r = RoleExpressionConverter((owlOAV.getProperty()));
			Formula c = null;
//			if(owlOAV.getFiller() instanceof OWLClass)
//				c = new AtomicConcept(((OWLClass)owlOAV.getFiller()).getIRI().toString());
//			else if(owlOAV.getFiller() instanceof OWLObjectComplementOfImpl){
//				OWLObjectComplementOfImpl coi = (OWLObjectComplementOfImpl) owlOAV.getFiller();
//				OWLClassExpression nn = coi.getNNF();
////				System.out.println("coi: "+coi);
//				System.out.println("nn: "+nn.getComplementNNF());
//				if(nn.getComplementNNF()  instanceof OWLClass){
//					c = new AtomicConcept("\u00AC"+((OWLClass) nn.getComplementNNF()).getIRI().toString());
//					c.setNeg();
//				}else {
//					return ALCClassExpressionConverter(nn);
//				}
//			}else{
				c = ALCClassExpressionConverter(owlOAV.getFiller());
//			}
			if(r == null || c == null) return null;
			return new Forall(r,c);
		}
		else if(concept instanceof OWLDataHasValue){
			OWLDataHasValue owlDHV = (OWLDataHasValue) concept;
			DataProperty r = new DataProperty(((OWLDataProperty)owlDHV.getProperty()).getIRI().toString());
			Literal c = new Literal((owlDHV.getFiller().getLiteral()));
			if(r == null || c == null) return null;
			return new HasValue(r,c);
		}else if(concept instanceof  OWLObjectComplementOf){
			OWLObjectComplementOf owlOC = (OWLObjectComplementOf) concept;
			OWLClassExpression nn = owlOC.getNNF();
			System.out.println("nn: "+nn);
			if(nn.getComplementNNF() instanceof OWLClass){
				Formula c = new AtomicConcept("\u00AC"+((OWLClass) nn.getComplementNNF()).getIRI().toString());
				c.setNeg();
				return c;
			}else {
				Formula cs = ALCClassExpressionConverter(nn.getComplementNNF());
				if(cs==null) return null;
				Formula c = new AtomicConcept("\u00AC"+cs.getText());
				return c;
			}
		}
		return null;
		//return TopConcept.getInstance();
	}

	public List<Formula> ABoxConvertor(OWLOntology ontology){
		List<Formula> formula_list = new ArrayList<>();

		for (OWLNamedIndividual i: ontology.getIndividualsInSignature()) {
			for(OWLClassAssertionAxiom owlAxiom:ontology.getClassAssertionAxioms(i))
				if((owlAxiom.getClassExpression() instanceof OWLClass)) // ToDO: add concept assertion for complex concept
					formula_list.add(new ConceptAssertion(new AtomicConcept(((OWLClass)owlAxiom.getClassExpression()).getIRI().toString()), new Individual(i.getIRI().toString())));
			for(OWLObjectPropertyAssertionAxiom owlAxiom:ontology.getObjectPropertyAssertionAxioms(i))
				formula_list.add(new RoleAssertion(new AtomicRole(((OWLObjectProperty)owlAxiom.getProperty()).getIRI().toString()), new Individual(((OWLNamedIndividual)owlAxiom.getSubject()).getIRI().toString()), new Individual(((OWLNamedIndividual)owlAxiom.getObject()).getIRI().toString())));
		}
		return formula_list;
	}
	private RoleExpression RoleExpressionConverter(OWLPropertyExpression role) {

		if (role instanceof OWLObjectProperty) {
			OWLObjectProperty owlOP = (OWLObjectProperty) role;
//			if(objectMap.containsKey(owlOP.getIRI().toString())==false)
//				objectMap.put(owlOP.getIRI().toString(), new AtomicRole(owlOP.getIRI().toString()));
			return new AtomicRole(owlOP.getIRI().toString());//(RoleExpression) objectMap.get(owlOP.getIRI().toString());
			
		}
		return null;
		//return TopRole.getInstance();
	}

	public List<Formula> ABoxNormalize(OWLOntology ontology){
		List<Formula> formula_list = new ArrayList<>();

		for (OWLNamedIndividual i: ontology.getIndividualsInSignature()) {
			for(OWLClassAssertionAxiom owlAxiom:ontology.getClassAssertionAxioms(i))
				if((owlAxiom.getClassExpression() instanceof OWLClass)) // ToDO: add concept assertion for complex concept
					formula_list.add(ConceptAssertionNormalize(new AtomicConcept(((OWLClass)owlAxiom.getClassExpression()).getIRI().toString()), new AtomicConcept(i.getIRI().toString())));
			for(OWLObjectPropertyAssertionAxiom owlAxiom:ontology.getObjectPropertyAssertionAxioms(i))
				formula_list.add(RoleAssertionNormalize(new AtomicRole(((OWLObjectProperty)owlAxiom.getProperty()).getIRI().toString()), new AtomicConcept(((OWLNamedIndividual)owlAxiom.getObject()).getIRI().toString()), new AtomicConcept(((OWLNamedIndividual)owlAxiom.getSubject()).getIRI().toString())));
		}
		return formula_list;
	}

	public Formula ConceptAssertionNormalize(AtomicConcept c, AtomicConcept i){
		return new Inclusion(i,c);
	}

	public Formula RoleAssertionNormalize(AtomicRole r, AtomicConcept c1, AtomicConcept c2){
		return new Inclusion(c1, new Exists(r,c2));
	}
//	public static  void main(String [] args){
//
//		Set<Formula> now = new LinkedHashSet<>();
//		Set<Formula> now2 =new LinkedHashSet<>();
//		now2.add(new AtomicConcept("A"));
//		now2.add(new AtomicConcept("B"));
//		Formula and =  And.getAnd(now2);
//		Set<Formula> temp = new HashSet<>();
//		temp.add(and);
//		temp.add(new AtomicConcept("H"));
//		now.add(new Exists(new AtomicRole("r"), And.getAnd(temp)));
//		now.add(new AtomicConcept("C"));
//		System.out.println(now);
//
//		Formula temp2 = new Converter().RightSubformulaConverter( And.getAnd(now));
//		System.out.println(temp2);
//		System.out.println(temp2.getSubformulae());
//		for(Formula t : temp2.getSubformulae()){
//			if(t instanceof Exists){
//				System.out.println(t.getSubFormulas().get(1).getSubformulae());
//			}
//		}
//	}

}
