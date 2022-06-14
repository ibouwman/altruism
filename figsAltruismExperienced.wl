(* SET PARAMETERS HERE *)

(* path for input and output:*)
SetDirectory["/home/hermsen/hermsen2/Simulations/Cooperation/test35_final_default_30_other_Seed_2"];
(* linear size of the grid: *)
res = 1024;
(* start with the frame with this number: *)
from = 1;
(* end with the frame with this number *)
to = 2001;
(* scale the cooperation
	such that concentrations above this value
	saturate the color scale *)
maxvalue = 5.5;
sigCoop = 10;(*scale of cooperation. *)
sigComp = 40;


(* RUN! *)
ParallelDo[
	(
	Print[i];
    ticksLeft = Table[{i*sigCoop, ""}, {i, 0, Floor[res/sigCoop]}];
    ticksRight = Table[{i*sigComp, ""}, {i, 0, Floor[res/sigComp]}];
	plot = ArrayPlot[
	(Import["coop" <> IntegerString[i, 10, 4] <> ".txt" , "Table"])/
     maxvalue,
     ColorFunctionScaling -> False,
     Frame -> True,
     FrameTicks -> {{ticksLeft, ticksRight}, {None, None}},
     PlotRangePadding->0,
     ImageSize -> {res, res},
     PixelConstrained->True,
     ColorFunction -> (ColorData["SouthwestColors"][#1] &)(*,
     MaxPlotPoints -> Infinity*)
    ];
	Export[
		("alt" <> IntegerString[i, 10, 4] <> ".jpg"),
		plot,
		"CompressionLevel" -> Automatic
		];
	), {i, from, to}];







