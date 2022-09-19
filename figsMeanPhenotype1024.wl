(* ::Package:: *)

(* SET PARAMETERS HERE *)

(* path for input and output:*)
SetDirectory["output_production"];
(* linear size of the grid: *)
res = 1024;
(* start with the frame with this number: *)
from = 1;
(* end with the frame with this number *)
to = 81; 
(* scale the phenotype
	such that concentrations above this value
	saturate the color scale *)
maxvalue = .025;
(* scale the frame number to get the actual time *)
scaletime = 0.08*625;  (* one timestep is 0.08 time units, we generate output every 50 timesteps*)
sigCoop = 10; (* scale of cooperation in grid cells. *)
sigComp = 40; (* scale of competition *) 

(* WE USE THIS COLOR SCALE *)


(*
The value 0 is reserved for empty squares.
Below, the color function is defined such that squares with a value below a very small "cutoff" become white.
However, it is possible that a cell contains organisms with trait value below the cutoff;
we want those cells to get a color. 
We therefore add "cutoff" to the actual value.
 *)
cutoff = 10^(-12);
(*
The fontsize and font family in the legend *)
fontsize = 24;
font = "Arial";
(* colorfunction *)
mycolorfunction = Quiet[Blend[{{0.`,RGBColor[1,1,1]},{cutoff, RGBColor[0.39681100000193564`, 0.3101399999997449, 0.20410499999951026`]},{66666666667/400000000000,RGBColor[0.6427400000004629, 0.33057700000073176`, 0.15407550000006578`]},{166666666667/500000000000,RGBColor[0.7267320000003851, 0.5381360000008305, 0.3159300000007028]},{1000000000001/2000000000000,RGBColor[0.8178820000002428, 0.7260905000004458, 0.42699099999994533`]},{666666666667/1000000000000,RGBColor[0.8319639999996785, 0.8105429999998334, 0.37285399999970387`]},{1666666666667/2000000000000,RGBColor[0.6419974999997516, 0.71831849999989, 0.3669065000002783]},{1,RGBColor[0.35082, 0.595178, 0.853742]}},#1]]&

labels = Directive[
	FontSize -> fontsize,
	FontFamily -> font
	]

legend = BarLegend[
  {
   "SouthwestColors", {0, maxvalue}},
  LabelStyle -> labels ,
  LegendLabel -> "phenotype",
  Method -> {
		Frame -> False,
		TicksStyle -> Directive[Black, Thick]
    },
  LegendMarkerSize -> 500
  ]


(* RUN! *)
ParallelDo[(
	Print[i];
    t = Round[(i-1)*scaletime];
	occs = Import["200073_density_" <> IntegerString[i, 10, 4] <> ".txt" , "Table"];
	prod = Import["200073_sumaltr_" <> IntegerString[i, 10, 4] <> ".txt" , "Table"];
	meanprod = prod;
	For[k=1, k<res + 1, k++,
		For[j = 1, j< res + 1, j++,
			If[ occs[[k,j]] > 0,
				meanprod[[k,j]] = prod[[k,j]]/occs[[k,j]] + 1.1*cutoff,
				meanprod[[k,j]] = 0
				]
			]
		];
	ticksLeft = Table[{i*sigCoop, ""}, {i, 0, Floor[res/sigCoop]}];
	ticksRight = Table[{i*sigComp, ""}, {i, 0, Floor[res/sigComp]}];
	plot = ArrayPlot[
		meanprod/maxvalue,
		ColorFunctionScaling->False,
		Frame->True,
		ImageSize->{res,res},
		PixelConstrained -> True,
		PlotRangePadding -> 0,
		ColorFunction -> mycolorfunction,
		FrameTicks -> {{ticksLeft, ticksRight}, {None, None}},
        Epilog -> {
			Text[
				Style[
					"t = " <> IntegerString[t],
					FontSize-> fontsize,
					FontFamily-> font
					],
				Scaled[{.9, .05}]
				]
			}
        (*,
		MaxPlotPoints -> Infinity
		*)
		];
	Export[
		("200073_meanaltr" <> IntegerString[i, 10, 4] <> ".jpg"),
		plot, (* Legended[plot, Placed[legend, After, Identity]], *)
		"CompressionLevel" -> Automatic
		];
	),
	{ i,from, to}]
