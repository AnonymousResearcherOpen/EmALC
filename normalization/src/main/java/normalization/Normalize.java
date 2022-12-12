package normalization;

import com.sun.xml.internal.bind.v2.schemagen.xmlschema.Union;
import concepts.AtomicConcept;
import connectives.*;
import formula.Formula;
import roles.AtomicRole;

import java.text.Normalizer;
import java.util.*;

public class Normalize {
    public DefinerIntroducer introducedDefiners;
    public Map<Formula, AtomicConcept> definer_left_map = new HashMap<>();
    public  Map<Formula, AtomicConcept> definer_right_map = new HashMap<>();
    public  Map<AtomicConcept, Formula> reverse_definer_left_map = new HashMap<>();
    public  Map<AtomicConcept, Formula> reverse_definer_right_map = new HashMap<>();
    public Set<AtomicConcept> definers = new HashSet<>();
    public Normalize(){
        introducedDefiners = new DefinerIntroducer();
    }
    public boolean andOr(Formula f){
        if((f instanceof And) || (f instanceof  Or)) return true;
        return false;
    }
    public boolean existForall(Formula f){
        if((f instanceof Exists) || (f instanceof Forall)) return true;
        return false;
    }
    public boolean isCompound(Formula f){
        if(andOr(f)){
            List<Formula> a1 = f.getSubFormulas();
            if(a1.size()>2) return true;
            if(a1.size()==2) {
                if (a1.get(0).getSubFormulas() != null) return true;
                if (a1.get(1).getSubFormulas() != null) return true;
                return false;
            }
            return false;
        }else if (existForall(f)){
            Formula a2 = f.getSubFormulas().get(1);
            if(a2.getSubFormulas()==null) {
                return false;
            }
            return true;
        }
        return false;
    }
    public int getAxiomType(Formula left_hand, Formula right_hand){
        boolean f1 = isCompound(left_hand);
        boolean f2 = isCompound(right_hand);
        if(!f1 && !f2){
            if(existForall(left_hand)||andOr(left_hand)){
                if(existForall(right_hand)||andOr(right_hand)){
                    return 2;
                }
            }
            if(left_hand instanceof Or) return 3;
            if(right_hand instanceof And) return 4;
            return 1;
        }
        else if(f1 && f2) return 2;
        else if(f1){
            return 3;
        }
        return 4;

    }
    public List<Formula> getELFormulaNormalization(Formula f){

        List<Formula> fs = new ArrayList<>();
        Formula left_hand = f.getSubFormulas().get(0);
        Formula right_hand = f.getSubFormulas().get(1);
        List<Formula> left_hand_formulas = left_hand.getSubFormulas();
        List<Formula> right_hand_formulas = right_hand.getSubFormulas();
        int axiom_type = getAxiomType(left_hand,right_hand);
        if(axiom_type==1) return Collections.singletonList(f);
        else if(axiom_type==2){
            AtomicConcept definer_c = introducedDefiners.introduceDefiner();
            fs.addAll(getELFormulaNormalization(new Inclusion(left_hand, definer_c)));
            fs.addAll(getELFormulaNormalization(new Inclusion(definer_c, right_hand)));
        }
        else if(axiom_type==3){
            if(left_hand instanceof And){
                Formula a1 = left_hand_formulas.get(0);
                Formula a2 = left_hand_formulas.get(1);
                axiom_type = getAxiomType(a1,a2);
                if(axiom_type==3){
                    AtomicConcept definer_c = introducedDefiners.introduceDefiner();
                    fs.addAll(getELFormulaNormalization(new Inclusion(a1, definer_c)));
                    Set<Formula> ands = new HashSet<>(); ands.add(definer_c); ands.add(a2);
                    fs.addAll(getELFormulaNormalization(new Inclusion(And.getAnd(ands), left_hand)));
                }
                if(axiom_type==4){
                    AtomicConcept definer_c = introducedDefiners.introduceDefiner();
                    fs.addAll(getELFormulaNormalization(new Inclusion(a2, definer_c)));
                    Set<Formula> ands = new HashSet<>(); ands.add(definer_c); ands.add(a1);
                    fs.addAll(getELFormulaNormalization(new Inclusion(And.getAnd(ands), left_hand)));
                }
            }else if(left_hand instanceof Exists){
                Formula r = left_hand_formulas.get(0);
                Formula a = left_hand_formulas.get(1);
                AtomicConcept definer_c = introducedDefiners.introduceDefiner();
                fs.addAll(getELFormulaNormalization(new Inclusion(a, definer_c)));
                fs.addAll(getELFormulaNormalization(new Inclusion(new Exists(r,definer_c), right_hand)));
            }
        }
        else if(axiom_type==4){
            if(right_hand instanceof And){
                Formula a1 = right_hand_formulas.get(0);
                Formula a2 = right_hand_formulas.get(1);
                fs.addAll(getELFormulaNormalization(new Inclusion(left_hand, a1)));
                fs.addAll(getELFormulaNormalization(new Inclusion(left_hand, a2)));
            }else{
                Formula r = right_hand_formulas.get(0);
                Formula a = right_hand_formulas.get(1);
                AtomicConcept definer_c = introducedDefiners.introduceDefiner();
                fs.addAll(getELFormulaNormalization(new Inclusion(definer_c, a)));
                fs.addAll(getELFormulaNormalization(new Inclusion(left_hand, new Exists(r,definer_c))));
            }

        }

        return fs;
    }
    public List<Formula> ELNormalize(List<Formula> o){
        List<Formula> o_ = new ArrayList<>();
        for(Formula f:o){
            o_.addAll(getELFormulaNormalization(f));
        }
        return o_;
    }

    public List<Formula> getALCFormulaNormalization(Formula f){
        boolean dual = false;
        if(f instanceof Equivalence) {dual = true;System.out.println("Equivalence: "+f);}
        List<Formula> fs = new ArrayList<>();
        Formula left_hand = f.getSubFormulas().get(0);
        Formula right_hand = f.getSubFormulas().get(1);
        List<Formula> left_hand_formulas = new ArrayList<>(); List<Formula> right_hand_formulas = new ArrayList<>();
        if((Objects.isNull(left_hand)==false) &&(Objects.isNull(left_hand.getSubFormulas())==false)) left_hand_formulas.addAll(left_hand.getSubFormulas());
//        System.out.println("right_hand: "+right_hand);
        if((Objects.isNull(right_hand)==false) && (Objects.isNull(right_hand.getSubFormulas()) == false)) right_hand_formulas.addAll(right_hand.getSubFormulas());
        int axiom_type = getAxiomType(left_hand,right_hand);
        if(axiom_type==1){
            if(dual){
                fs.add(new Inclusion(right_hand, left_hand));
                fs.add(new Inclusion(left_hand, right_hand));
                return fs;
            }
            return Collections.singletonList(f);
        }
        else if(axiom_type==2){
            AtomicConcept definer_c = introducedDefiners.introduceDefiner();
            Formula f1; Formula f2;
            if(dual){
                f1 = new Equivalence(left_hand, definer_c);
                f2 = new Equivalence(definer_c, right_hand);
            }else{
                f1 = new Inclusion(left_hand, definer_c);
                f2 = new Inclusion(definer_c, right_hand);
            }
            fs.addAll(getALCFormulaNormalization(f1));
            fs.addAll(getALCFormulaNormalization(f2));
        }
        else if(axiom_type==3){
            if(left_hand instanceof And){
                int idx = left_hand_formulas.size();
                for(int i=0;i<left_hand_formulas.size();i++){
                    Formula ff = left_hand_formulas.get(i);
                    if(ff.getSubFormulas()==null){
                        idx = i;
                        continue;
                    }
                }

                Formula a1;
                if(idx==left_hand_formulas.size()){
                    AtomicConcept definer_c1 = introducedDefiners.introduceDefiner();AtomicConcept definer_c2 = introducedDefiners.introduceDefiner();
                    Set<Formula> ands = new HashSet<>(); ands.add(definer_c2);ands.add(definer_c1);
                    Set<Formula> ands2 = new HashSet<>(); ands2.addAll(left_hand_formulas.subList(1,left_hand_formulas.size()));
                    Formula af = ands2.iterator().next();
                    if(ands.size()>1) af = And.getAnd(ands2);

                    Formula f1; Formula f2; Formula f3;
                    if(dual){
                        f1 = new Equivalence(And.getAnd(ands), right_hand);
                        f2 = new Equivalence(definer_c1, left_hand_formulas.get(0));
                        f3 = new Equivalence(definer_c2, af);
                    }else{
                        f1 = new Inclusion(And.getAnd(ands), right_hand);
                        f2 = new Inclusion(definer_c1, left_hand_formulas.get(0));
                        f3 = new Inclusion(definer_c2, af);
                    }

                    fs.addAll(getALCFormulaNormalization(f1));
                    fs.addAll(getALCFormulaNormalization(f2));
                    fs.addAll(getALCFormulaNormalization(f3));
                }else{
                    a1 = left_hand_formulas.get(idx);
                    Set<Formula> sub = new HashSet<>(left_hand_formulas.subList(idx+1,left_hand_formulas.size()));
                    sub.addAll(left_hand_formulas.subList(0,idx));
                    Formula a2 = sub.iterator().next();
                    if(sub.size()>1)  a2 = And.getAnd(sub);

                    AtomicConcept definer_c = introducedDefiners.introduceDefiner();
                    Set<Formula> ands = new HashSet<>(); ands.add(definer_c); ands.add(a1);

                    Formula af = ands.iterator().next();
                    if(ands.size()>1) af = And.getAnd(ands);

                    Formula f1; Formula f2;
                    if(dual){
                        f1 = new Equivalence(a2, definer_c);
                        f2 = new Equivalence(af, right_hand);
                    }else{
                        f1 = new Inclusion(a2, definer_c);
                        f2 = new Inclusion(af, right_hand);
                    }
                    fs.addAll(getALCFormulaNormalization(f1));
                    fs.addAll(getALCFormulaNormalization(f2));
                }

            }else if(left_hand instanceof Exists){
                Formula r = left_hand_formulas.get(0);
                Formula a = left_hand_formulas.get(1);
                AtomicConcept definer_c = introducedDefiners.introduceDefiner();
                Formula f1; Formula f2;
                if(dual){
                    f1 = new Equivalence(a, definer_c);
                    f2 = new Equivalence(new Exists(r,definer_c), right_hand);
                }else{
                    f1 = new Inclusion(a, definer_c);
                    f2 = new Inclusion(new Exists(r,definer_c), right_hand);
                }
                fs.addAll(getALCFormulaNormalization(f1));
                fs.addAll(getALCFormulaNormalization(f2));
            }else if(left_hand instanceof Forall){
                Formula r = left_hand_formulas.get(0);
                Formula a = left_hand_formulas.get(1);
                AtomicConcept definer_c = introducedDefiners.introduceDefiner();
                Formula f1; Formula f2;
                if(dual){
                    f1 = new Equivalence(a, definer_c);
                    f2 = new Equivalence(new Forall(r,definer_c), right_hand);
                }else{
                    f1 = new Inclusion(a, definer_c);
                    f2 = new Inclusion(new Forall(r,definer_c), right_hand);
                }
                fs.addAll(getALCFormulaNormalization(f1));
                fs.addAll(getALCFormulaNormalization(f2));
            }else if(left_hand instanceof Or){
                for(Formula a:left_hand_formulas){
                    if(dual){
                        fs.addAll(getALCFormulaNormalization(new Equivalence(a, right_hand)));
                    }else{
                        fs.addAll(getALCFormulaNormalization(new Inclusion(a, right_hand)));
                    }

                }
            }
        }
        else if(axiom_type==4){
//            System.out.println("left: "+left_hand);
//            System.out.println("right: "+right_hand);
            if(right_hand instanceof Or){
                int idx = right_hand_formulas.size();
                for(int i=0;i<right_hand_formulas.size();i++){
                    Formula ff = right_hand_formulas.get(i);
                    if(ff.getSubFormulas()==null){
//                        System.out.println("ff: "+ff);
                        idx = i;continue;
                    }
                }
                Formula a1;
                if(idx == right_hand_formulas.size()){
                    AtomicConcept definer_c1 = introducedDefiners.introduceDefiner();AtomicConcept definer_c2 = introducedDefiners.introduceDefiner();
                    Set<Formula> ors = new HashSet<>(); ors.add(definer_c2);ors.add(definer_c1);
                    Formula of = ors.iterator().next();
                    if(ors.size()>1) of = Or.getOr(ors);

                    Set<Formula> ors2 = new HashSet<>(); ors2.addAll(right_hand_formulas.subList(1,right_hand_formulas.size()));
                    of = ors2.iterator().next();
                    if(ors2.size()>1) of = Or.getOr(ors2);

                    Formula f1; Formula f2; Formula f3;
                    if(dual){
                        f1 = new Equivalence(left_hand, of);
                        f2 = new Equivalence(right_hand_formulas.get(0), definer_c1);
                        f3 = new Equivalence(of, definer_c2);
                    }else{
                        f1 = new Inclusion(left_hand, of);
                        f2 = new Inclusion(right_hand_formulas.get(0), definer_c1);
                        f3 = new Inclusion(of, definer_c2);
                    }

                    fs.addAll(getALCFormulaNormalization(f1));
                    fs.addAll(getALCFormulaNormalization(f2));
                    fs.addAll(getALCFormulaNormalization(f3));
                }else{
                   a1 = right_hand_formulas.get(idx);
                    Set<Formula> sub = new HashSet<>(right_hand_formulas.subList(idx+1,right_hand_formulas.size()));
                    sub.addAll(right_hand_formulas.subList(0,idx));
                    Formula a2 = sub.iterator().next();
                    if(sub.size()>1) a2 = Or.getOr(sub);
//                    System.out.println("right_hand: "+right_hand_formulas);
//                    System.out.println("a1: "+a1);
//                    System.out.println("a2: "+a2);

                    AtomicConcept definer_c = introducedDefiners.introduceDefiner();
                    Set<Formula> ors = new HashSet<>();ors.add(a1);ors.add(definer_c);
                    Formula of = ors.iterator().next();
                    if(ors.size()>1) of = Or.getOr(ors);
                    Formula f1; Formula f2;
                    if(dual){
                        f1 = new Equivalence(left_hand, of);
                        f2 = new Equivalence(a2, definer_c);
                    }else{
                        f1 = new Inclusion(left_hand, of);
                        f2 = new Inclusion(a2, definer_c);
                    }
                    fs.addAll(getALCFormulaNormalization(f1));
                    fs.addAll(getALCFormulaNormalization(f2));
                }


            }
            else if(right_hand instanceof Exists){
                Formula r = right_hand_formulas.get(0);
                Formula a = right_hand_formulas.get(1);
                AtomicConcept definer_c = introducedDefiners.introduceDefiner();

                Formula f1; Formula f2;
                if(dual){
                    f1 = new Equivalence(definer_c, a);
                    f2 = new Equivalence(left_hand, new Exists(r,definer_c));
                }else{
                    f1 = new Inclusion(definer_c, a);
                    f2 = new Inclusion(left_hand, new Exists(r,definer_c));
                }

                fs.addAll(getALCFormulaNormalization(f1));
                fs.addAll(getALCFormulaNormalization(f2));
            }else if(right_hand instanceof Forall){
                Formula r = right_hand_formulas.get(0);
                Formula a = right_hand_formulas.get(1);
                AtomicConcept definer_c = introducedDefiners.introduceDefiner();

                Formula f1; Formula f2;
                if(dual){
                    f1 = new Equivalence(definer_c, a);
                    f2 = new Equivalence(left_hand, new Forall(r,definer_c));
                }else{
                    f1 = new Inclusion(definer_c, a);
                    f2 = new Inclusion(left_hand, new Forall(r,definer_c));
                }

                fs.addAll(getALCFormulaNormalization(f1));
                fs.addAll(getALCFormulaNormalization(f2));
            }else if(right_hand instanceof And){
                for(Formula a:right_hand_formulas){
                    if(dual){
                        fs.addAll(getALCFormulaNormalization(new Equivalence(left_hand, a)));
                    }else{
                        fs.addAll(getALCFormulaNormalization(new Inclusion(left_hand, a)));
                    }

                }
            }

        }

        return fs;
    }
    public List<Formula> ALCNormalize(List<Formula> o){
        List<Formula> o_ = new ArrayList<>();
        for(Formula f:o){
            o_.addAll(getALCFormulaNormalization(f));
        }
        return o_;
    }


}
