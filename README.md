# Genetic-Algorithm-with-Border-Trades

Python code to produce the paper _Genetic Algorithm with Border Trades (GAB)_: https://arxiv.org/abs/2501.18184

This project was based on the mlrose-hiive repository created by Hayes, G. and Rollings, A. at: https://github.com/hiive/mlrose. Additional genetic algorithm related code written or modified by me was all within /mlrose_hiive directory. Lines added or changed by me generally started with a comment: # added by Lyu, and completely with a comment: # edits end. For example:
	# added by Lyu
            new_start = child[0]
            last_end = next_gen[-1][-1]
            if new_start != last_end:
                next_gen.append(child)
            else:
                child = np.logical_not(child)
                next_gen.append(child)
            # edits end
	# added by Lyu
	 # edits end
Code Edits under /code/mlrose/mlrose_hiive/:
Flip Flop problems:
1. /algorithms/ga_border_check_front.py
	Modified from /algorithms/ga.py. 
2. /runners/ga-border-check_front_runner.py
	Modified from /runners/ga_runner.py. Changed names of algorithms accordingly.
3. __init__.py in /algorithms, /runners and ./
	imported additional classes and functions written in Step 1 and 2 above.
Job Scheduling problems - added new problem:
4. job_scheduling_opt.py in /opt_probs/
	initiated job scheduling problems
5. __init__.py in /fitness/
	initiated job scheduling fitness functions
6. job_scheduling.py in /fitness
	Defined fitness functions
	Turn on GAB-A1 or GAB-A2 in group_by_profit_duration() function
7. job_scheduling_generator.py in /generators/
	Created a job scheduling problem
8. __init__.py in /generators/, /algorithms/, /runners/ and ./
9. ga_js_border_check_simple_runner.py in /runners/
10. ga_js_border_check_simple.py in /algorithms/
	Turn on GAB-[A1/A2/B/C1/C2]



Analysis Notebooks: under /analysis/ directory

*Intermediate data (.csv files) was stored in /csv-files directory. File names should be self-explanatory. Please refer to .to_csv() commands in analysis notebooks below if you’re not sure.

*1. Flip Flop analysis
Available problem sizes: 7, 14, 28, 1,000

1a. with Simulated Annealing (SA)
FlipFlop-[problem size]-SA.ipynb analyzed flip flop problems with SA. For example, FlipFlop-7-SA.ipynb was for problem size 7.

1b. with the Canonical Genetic Algorithm (GA)
FlipFlop-GA-[problem size].ipynb. For example, FlipFlop-GA-7.ipynb analyzed Flip Flop problems of size 7 with GA 

1c. Genetic Algorithm with Border Trades (GAB)
FlipFlop-GABF-[problem size].ipynb. For example, FlipFlop-GABF-7.ipynb analyzed Flip Flop problems of size 7 with GAB

*2. Job Scheduling with Breaks analysis

2a. compare the performance of different border trade methods when problem size = 108
GA: js-ga-108.ipynb
GAB-A: js-GAB-mutate-tuned-A-108.ipynb
GAB-B: js-GAB-mutate-108.ipynb
GAB-C1: js-GAB-mutate-tuned-C1-108.ipynb
GAB-C2: js-GAB-mutate-tuned-C2-108.ipynb

2b. compare the performance of GA and GAB-B at various problem sizes
Available problem sizes: 3, 7, 10, 13, 18
js-GA-returned-[problem size].ipynb
js-GAB-mutate-[problem size].ipynb

2c. brute force method that verifies Job scheduling results
js-brute-force-verification.ipynb: a brute force evaluation is only applicable when problem sizes are small (< 10 tasks), due to the limit of runtime. For relative large task numbers, this notebook uses (Simulated Annealing) SA to verify convergence.

2d. fixed tasks generated randomly are stored in /txt-files/ directory
[problem size]_tasks.txt

2e. utility file
utility.py: generate random tasks with a fixed seed


@misc{lyu2025genetic,
  author = {Lyu, Qingchuan},
  title = {Genetic-Algorithm-with-Border-Trades},
  year = {2025},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/QingchuanLyu/Genetic-Algorithm-with-Border-Trades}},
  note = {Accessed: day month year}
}
