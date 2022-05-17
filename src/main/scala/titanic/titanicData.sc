import titanic.NaiveBayes.{calcPriorPropabilities, countAttributeValues, getAttributeValues}
import titanic.{NaiveBayes, TitanicDataSet, Utils}
import titanic.TitanicDataSet.{createDataSetForTraining, extractTrainingAttributes}

val train = Utils.loadDataCSV("train.csv")
//age kategorisieren
val dataattr = train.map(x => extractTrainingAttributes(x, List("sex", "age", "pclass")))
val pclassmean = (dataattr.map(x => x.filter(f => f._1 == "pclass")).flatMap(x => x).map(x => x._2.asInstanceOf[Int]).sum)/(dataattr.map(x => x.filter(f => f._1 == "pclass")).size)
val agemean = (dataattr.map(x => x.filter(f => f._1 == "age")).flatMap(x => x).map(x => x._2.asInstanceOf[Float]).sum)/(dataattr.map(x => x.filter(f => f._1 == "age")).size)
val tryy = dataattr.map(x => x.mapValues(y => x.getOrElse("age", agemean)))
TitanicDataSet.countAllMissingValues(tryy, List("sex","age","pclass"))



val classAttr = "survived"

val d = createDataSetForTraining(train)

val classVals= countAttributeValues(d,classAttr)

val aValues = getAttributeValues(d).asInstanceOf[ Map[String, Set[Any]]]

val prior= calcPriorPropabilities(d,classAttr)

val data: Map[Any, Set[(String, Map[Any, Int])]] = NaiveBayes.calcAttribValuesForEachClass(d,classAttr)

val cndProb = data.transform((z, y) => y.map(x => (x._1, aValues(x._1).zipWithIndex.toMap.transform((a, b) => (x._2.getOrElse(a, 0).asInstanceOf[Double] + 1.0) /(classVals(z.asInstanceOf[String].toInt) + aValues(x._1).size)))))
cndProb("1")


