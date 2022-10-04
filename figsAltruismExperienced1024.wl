(* SET PARAMETERS HERE *)

(* path for input and output:*)
SetDirectory["output10072"];
(* linear size of the grid: *)
res = 1024;
(* start with the frame with this number: *)
from = 1;
(* end with the frame with this number *)
to = 81;
(* scale the cooperation
	such that concentrations above this value
	saturate the color scale *)
maxvalue = 25;
sigCoop = 10;(*scale of cooperation. *)
sigComp = 40;


(* RUN! *)
ParallelDo[
	(
	Print[i];
    ticksLeft = Table[{i*sigCoop, ""}, {i, 0, Floor[res/sigCoop]}];
    ticksRight = Table[{i*sigComp, ""}, {i, 0, Floor[res/sigComp]}];
	plot = ArrayPlot[
	(Import["10072_expaltr_" <> IntegerString[i, 10, 4] <> ".txt" , "Table"])/
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
		("10072_expaltr" <> IntegerString[i, 10, 4] <> ".jpg"),
		plot,
		"CompressionLevel" -> Automatic
		];
	), {i, from, to}];







