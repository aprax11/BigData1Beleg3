import titanic.NaiveBayes

val trainDataSet = List(
  Map[String,String]("day"-> "weekday", "season"->"spring", "wind"->"none", "rain"->"none", "class"->"on time"),
  Map[String,String]("day"-> "weekday", "season"->"winter", "wind"->"none", "rain"->"slight", "class"->"on time"),
  Map[String,String]("day"-> "weekday", "season"->"winter", "wind"->"none", "rain"->"slight", "class"->"on time"),
  Map[String,String]("day"-> "weekday", "season"->"winter", "wind"->"high", "rain"->"heavy", "class"->"late"),
  Map[String,String]("day"-> "saturday", "season"->"summer", "wind"->"normal", "rain"->"none", "class"->"on time"),
  Map[String,String]("day"-> "weekday", "season"->"autumn", "wind"->"normal", "rain"->"none", "class"->"very late"),
  Map[String,String]("day"-> "holiday", "season"->"summer", "wind"->"high", "rain"->"slight", "class"->"on time"),
  Map[String,String]("day"-> "sunday", "season"->"summer", "wind"->"normal", "rain"->"none", "class"->"on time"),
  Map[String,String]("day"-> "weekday", "season"->"winter", "wind"->"high", "rain"->"heavy", "class"->"very late"),
  Map[String,String]("day"-> "weekday", "season"->"summer", "wind"->"none", "rain"->"slight", "class"->"on time"),
  Map[String,String]("day"-> "saturday", "season"->"spring", "wind"->"high", "rain"->"heavy", "class"->"cancled"),
  Map[String,String]("day"-> "weekday", "season"->"summer", "wind"->"high", "rain"->"slight", "class"->"on time"),
  Map[String,String]("day"-> "saturday", "season"->"winter", "wind"->"normal", "rain"->"none", "class"->"late"),
  Map[String,String]("day"-> "weekday", "season"->"summer", "wind"->"high", "rain"->"none", "class"->"on time"),
  Map[String,String]("day"-> "weekday", "season"->"winter", "wind"->"normal", "rain"->"heavy", "class"->"very late"),
  Map[String,String]("day"-> "saturday", "season"->"autumn", "wind"->"high", "rain"->"slight", "class"->"on time"),
  Map[String,String]("day"-> "weekday", "season"->"autumn", "wind"->"none", "rain"->"heavy", "class"->"on time"),
  Map[String,String]("day"-> "holiday", "season"->"spring", "wind"->"normal", "rain"->"slight", "class"->"on time"),
  Map[String,String]("day"-> "weekday", "season"->"spring", "wind"->"normal", "rain"->"none", "class"->"on time"),
  Map[String,String]("day"-> "weekday", "season"->"spring", "wind"->"normal", "rain"->"slight", "class"->"on time")
)

trainDataSet.groupBy(_.get("class")).transform((x,y) => y.flatMap(z => z.groupBy(_._1)).groupBy(_._1).transform((x,y) => y.map(_._2))).transform((x,y) => y.transform((x,y)=> y.flatten.groupBy(_._2).mapValues(_.size)).filter(_._1 != "class").toSet).map(x => (x._1.mkString, x._2))

val app= Map[Any,Double]("cancled"->0.0, "late"->0.0125, "on time"->0.0013, "very late"->0.0222)

app.toList.reduce((x,y) => if(x._2 > y._2) x else y)._1















val classVals= NaiveBayes.countAttributeValues(trainDataSet,"class")
val aValues = NaiveBayes.getAttributeValues(trainDataSet).asInstanceOf[ Map[String, Set[Any]]]
val data: Map[Any, Set[(String, Map[Any, Int])]] = NaiveBayes.calcAttribValuesForEachClass(trainDataSet,"class")

aValues.transform((x,y) => y.toList)


//data.transform((x,y)=> y.map(l => (l._1,l._2.updated()))
/*
data.transform((x,y) => y
  .map(y=> (y._1,y._2
    .transform((l,u) =>  (u+1).toDouble/(classVals(x) + y._2.size))
    )))

 */
val classVals= NaiveBayes.countAttributeValues(trainDataSet,"class")
val data= NaiveBayes.calcAttribValuesForEachClass(trainDataSet,"class")
val condProp = NaiveBayes.calcConditionalPropabilitiesForEachClass(data,classVals)
val prior= NaiveBayes.calcPriorPropabilities(trainDataSet,"class")
val el= Map[String,String]("day"->"weekday", "season"->"winter", "wind"->"high", "rain"->"heavy")
//val res= NaiveBayes.calcClassValuesForPrediction(el,condProp,prior)
val exp= List(("cancled",0.0), ("late",0.0125), ("on time",0.0013), ("very late",0.0222))

//get the important things
condProp.transform((x,y) => y.map(x => (x._1, x._2.filter(f => el(x._1) == f._1).getOrElse(el(x._1), 0.0))).toList.map(a => a._2).product * prior(x))

