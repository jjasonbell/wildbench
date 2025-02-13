<p align="center"><img src="logo.png" alt="Wild Bench" width="500"></p>

Wild Bench is a benchmark for tasks LLMs may encounter "in the wild".

A capable LLM might be rendered useless for some tasks if it cannot read a PDF. Since most benchmarks attempt to test LLMs in an apples-to-apples way, much of the input data is standardized. ARC is an interesting approach to testing LLMs using grids that are represented visually to humans, but ARC task data is passed to LLMs in JSON format. 

Wild Bench is meant to fill this gap by providing a **benchmark for _workflows_**. In some tasks, the PDF processing engine is being tested more than the LLM, which may be desirable if the PDF reader is the most important binding constraint.

The hope for Wild Bench is that it is at the very least interesting. Ideally, Wild Bench would help to cast more focus on bottlenecks in LLM workflows, surface good existing tools, and signal areas where innovations may be valuable.

The first iteration of Wild Bench is a proof-of-concept that uses naive, off-the-shelf methods. The scripts are minimally customized. In each case what is evaluated is approximately "whatever the person writing the task evaluation thought would be natural to reach for in this case." 
 
Wild Bench makes extensive use of edsl. https://github.com/expectedparrot/edsl
