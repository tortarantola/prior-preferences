
// DRIFT DIFFUSION MODEL: BAYESIAN w/BIASED PRIOR (UNBIASED DRIFT PARAMETER)

// Modified from code by Bradley Doll (https://github.com/dollbb/estRLParam)

data {
    int<lower=0> NS; // number of subjects
    int<lower=0> NP; // number of item pairs
    int<lower=0> NT; // number of trials
    int<lower=0, upper=1> correct[NS,NP,NT]; // vector of correct responses (0=wrong, 1=correct)
    int<lower=0, upper=1> feedback[NS,NP,NT]; // vector of feedback boxes (0=wrong answer, 1=correct answer)
    real bid_congruence[NS,NP,NT]; // bid for correct minus bid for incorrect item, out of 3.00 GBP
    real rt[NS,NP,NT]; // vector of response times in seconds (from PsychoPy recording, not eyelink)
    real rt_mins[NS]; // list of each subject"s lowest response time (except for subject C12, C26, and C31, which had implausibly low RTs, and for whom we"ve substituted the lowest RT from all other subjects)
}

parameters {
	
	// DRIFT INTERCEPTS
	// Hyper-parameters
	real<lower=0, upper=3> threshold_int_mean; // Group mean of threshold intercept
	real<lower=0, upper=1> threshold_int_sd; // Pre-transform group standard deviation
	// Subject-level
	real<lower=0> threshold_int[NS];

	// DRIFT LEARNING WEIGHTS
	// Hyper-parameters
	real<lower=-3, upper=10> drift_rate_learning_mean; // Group mean of learning coefficient for drift rate
	real<lower=0, upper=pi()/2> drift_rate_learning_sd_unif; // Group standard deviation
	// Subject-level
	vector[NS] drift_rate_learning_raw;


	// DRIFT CONGRUENCE WEIGHT
	// Hyper-parameters
	real<lower=-5, upper=5> cong_weight_mean; // Group mean of preference congruence bias weight
	real<lower=0, upper=pi()/2> cong_weight_sd_unif; // Pre-transform group standard deviation of preference congruence bias weight
	// Subject-level
	vector[NS] cong_weight_raw;

	// NON-DECISION TIMES
	real<lower=0, upper=rt_mins[1]> ndt_1;
	real<lower=0, upper=rt_mins[2]> ndt_2;
	real<lower=0, upper=rt_mins[3]> ndt_3;
	real<lower=0, upper=rt_mins[4]> ndt_4;
	real<lower=0, upper=rt_mins[5]> ndt_5;
	real<lower=0, upper=rt_mins[6]> ndt_6;
	real<lower=0, upper=rt_mins[7]> ndt_7;
	real<lower=0, upper=rt_mins[8]> ndt_8;
	real<lower=0, upper=rt_mins[9]> ndt_9;
	real<lower=0, upper=rt_mins[10]> ndt_10;
	real<lower=0, upper=rt_mins[11]> ndt_11;
	real<lower=0, upper=rt_mins[12]> ndt_12;
	real<lower=0, upper=rt_mins[13]> ndt_13;
	real<lower=0, upper=rt_mins[14]> ndt_14;
	real<lower=0, upper=rt_mins[15]> ndt_15;
	real<lower=0, upper=rt_mins[16]> ndt_16;
	real<lower=0, upper=rt_mins[17]> ndt_17;
	real<lower=0, upper=rt_mins[18]> ndt_18;
	real<lower=0, upper=rt_mins[19]> ndt_19;
	real<lower=0, upper=rt_mins[20]> ndt_20;
	real<lower=0, upper=rt_mins[21]> ndt_21;
	real<lower=0, upper=rt_mins[22]> ndt_22;
	real<lower=0, upper=rt_mins[23]> ndt_23;
	real<lower=0, upper=rt_mins[24]> ndt_24;
	real<lower=0, upper=rt_mins[25]> ndt_25;
	real<lower=0, upper=rt_mins[26]> ndt_26;
	real<lower=0, upper=rt_mins[27]> ndt_27;
	real<lower=0, upper=rt_mins[28]> ndt_28;
	real<lower=0, upper=rt_mins[29]> ndt_29;
	real<lower=0, upper=rt_mins[30]> ndt_30;
}

transformed parameters {
	real drift_rate_learning_sd;
	real<lower=0> cong_weight_sd;
	vector[NS] drift_rate_learning;
	vector[NS] cong_weight;
	real threshold_int_a;
	real threshold_int_b;
	real<lower=0> non_decision_time_int[NS]; 

 
 	// Normally distributed priors
 	drift_rate_learning_sd <- 0 + 0.5*tan(drift_rate_learning_sd_unif);
 	cong_weight_sd <- 0 + 2.5*tan(cong_weight_sd_unif);

 	drift_rate_learning <- drift_rate_learning_mean + drift_rate_learning_sd * drift_rate_learning_raw;
 	cong_weight <- cong_weight_mean + cong_weight_sd * cong_weight_raw;

 	// Gamma distributed priors
 	threshold_int_a <- pow((threshold_int_mean / threshold_int_sd),2); // Shape parameter of threshold intercept
 	threshold_int_b <- threshold_int_mean / pow(threshold_int_sd, 2); // Rate (inverse scale) parameter of threshold intercept

 	// Non-decision time vector
 	non_decision_time_int[1] <- ndt_1;
 	non_decision_time_int[2] <- ndt_2;
 	non_decision_time_int[3] <- ndt_3;
 	non_decision_time_int[4] <- ndt_4;
 	non_decision_time_int[5] <- ndt_5;
 	non_decision_time_int[6] <- ndt_6;
 	non_decision_time_int[7] <- ndt_7;
 	non_decision_time_int[8] <- ndt_8;
 	non_decision_time_int[9] <- ndt_9;
 	non_decision_time_int[10] <- ndt_10;
 	non_decision_time_int[11] <- ndt_11;
 	non_decision_time_int[12] <- ndt_12;
 	non_decision_time_int[13] <- ndt_13;
 	non_decision_time_int[14] <- ndt_14;
 	non_decision_time_int[15] <- ndt_15;
 	non_decision_time_int[16] <- ndt_16;
 	non_decision_time_int[17] <- ndt_17;
 	non_decision_time_int[18] <- ndt_18;
 	non_decision_time_int[19] <- ndt_19;
 	non_decision_time_int[20] <- ndt_20;
 	non_decision_time_int[21] <- ndt_21;
 	non_decision_time_int[22] <- ndt_22;
 	non_decision_time_int[23] <- ndt_23;
 	non_decision_time_int[24] <- ndt_24;
 	non_decision_time_int[25] <- ndt_25;
 	non_decision_time_int[26] <- ndt_26;
 	non_decision_time_int[27] <- ndt_27;
 	non_decision_time_int[28] <- ndt_28;
 	non_decision_time_int[29] <- ndt_29;
 	non_decision_time_int[30] <- ndt_30;
}

model {
	vector[2] q; // set up a 2-item array to hold two probability values, one for the correct and one for the incorrect answer
	vector[2] bid_cong; // vector to hold bid congruence to calculate softmax for prior
	real pri; // set up variable to hold PRIOR PERCEIVED probability of correct answer being correct
	real pr; // set up variable to hold PERCEIVED probability of correct answer being correct
	real a; // placeholder variable for threshold
	real ti; // non-decision time
	real v; // drift rate

	real p_correct; // placeholder variable for MODEL'S probability of a correct response

	// Convenience variables for Bayes rule
    real pr_d_c; // p(data | correct)
    real pr_d_not_c; // p(data | ~correct)
    int cf; // cumulative correct feedback so far for this pair

	// Hyper-parameter priors
	threshold_int_mean ~ normal(1,10);
	threshold_int_sd ~ uniform(0,2);

	drift_rate_learning_mean ~ normal(0,20);
	drift_rate_learning_sd_unif ~ uniform(0,pi()/2);

	cong_weight_mean ~ normal(0, 5);
	cong_weight_sd_unif ~ uniform(0,pi()/2);

	// Subject-level priors
	threshold_int ~ gamma(threshold_int_a, threshold_int_b);
	drift_rate_learning_raw ~ normal(0, 1);
	cong_weight_raw ~ normal(0, 1);
	
	for (s in 1:NS) {

		// Priors for non-decision time
		non_decision_time_int[s] ~ uniform(0, rt_mins[s]); // Non-hierarchical priors for non-decision times, with the upper bound being the lowest RT for that subject
		
		for (p in 1:NP) {
			
			cf <- 0;
			bid_cong[1] <- bid_congruence[s,p,1];
			bid_cong[2] <- 0;
			pri <- softmax(cong_weight[s] * bid_cong)[1]; // prior is a softmax of the bid difference with inverse temperature cong_weight[s]
			q[2] <- pri; // perceived probability of CORRECT answer being correct
			q[1] <- (1 - pri); // perceived probability of WRONG answer being correct

			for (t in 1:NT) {
				a <- threshold_int[s];
				ti <- non_decision_time_int[s];
				v <- drift_rate_learning[s] * ( q[2] - q[1] ) ;

// 				if (v == 0) {
// 					p_correct <- 0.5; 
// 				}
// 				else {
// 					p_correct <- 1 - ( (1 - exp(-v*a)) / ( exp(v*a) - exp(-v*a) ) );
// 				}
// //print ("v: ",v," a: ",a," p_correct: ",p_correct);
// 				// Sampling statement for choice data
// 				correct[s,p,t] ~ bernoulli(p_correct); 

				// Sampling statement for response time data
				if ( rt[s,p,t] > 0.41 ) { // Lowest plausible RT in data (excluding implausible RTs)

					if (correct[s,p,t] == 1) {
						rt[s,p,t] ~ wiener(a, ti, 0.5, v); // For correct responses, return upper boundary
					}
					else {
						rt[s,p,t] ~ wiener(a, ti, 0.5, -v); // For incorrect responses, return lower boundary
					}
				}
				cf <- cf + feedback[s,p,t]; // increase cumulative feedback by feedback on this trial
                pr_d_c <- exp(binomial_log(cf,t,0.8));
                pr_d_not_c <- exp(binomial_log(cf,t,0.2));
                pr <- (pr_d_c * pri) / ( (pr_d_c * pri) + (pr_d_not_c * (1-pri)) ); // Perceived probability of correct answer being correct, given cumulative feedback
				q[2] <- pr;
				q[1] <- (1 - pr);
			}
		}
	}
}

generated quantities {
	vector[NS*NP*NT] log_lik_rt; 
	vector[NS*NP*NT] log_lik_resp;
	int ix;
	vector[2] q; // set up a 2-item array to hold two Q values, one for the correct and one for the incorrect answer
	vector[2] bid_cong; // vector to hold bid congruence to calculate softmax for prior
	real pri; // set up variable to hold PRIOR PERCEIVED probability of correct answer being correct
	real pr; // set up variable to hold PERCEIVED probability of correct answer being correct
	real a; // placeholder variable for threshold
	real ti; // non-decision time
	real v; // drift rate

	real p_correct; // placeholder variable for probability of a correct response

	// Convenience variables for Bayes rule
    real pr_d_c; // p(data | correct)
    real pr_d_not_c; // p(data | ~correct)
    int cf; // cumulative correct feedback so far for this pair

	for (s in 1:NS) {
		
		for (p in 1:NP) {
			
			cf <- 0;
			bid_cong[1] <- bid_congruence[s,p,1];
			bid_cong[2] <- 0;
			pri <- softmax(cong_weight[s] * bid_cong)[1]; // prior is a softmax of the bid difference with inverse temperature cong_weight[s]
			q[2] <- pri; // perceived probability of CORRECT answer being correct
			q[1] <- (1 - pri); // perceived probability of WRONG answer being correct

			for (t in 1:NT) {
				ix <- (s-1)*NP*NT + (p-1)*NT + t; // index of log_lik_rt and lik_resp vectors

				a <- threshold_int[s];
				ti <- non_decision_time_int[s];
				v <- drift_rate_learning[s] * ( q[2] - q[1] ) ;

				if (v == 0) {
					p_correct <- 0.5; 
				}
				else {
					p_correct <- 1 - ( (1 - exp(-v*a)) / ( exp(v*a) - exp(-v*a) ) );
				}


				if (correct[s,p,t] == 1) {
					if (rt[s,p,t] >  -1) { // If RT data are not missing for that trial
						log_lik_rt[ix] <- wiener_log(rt[s,p,t], a, ti, 0.5, v); // For correct responses, return upper boundary
					}
					log_lik_resp[ix] <- log(p_correct); // Log probability of being correct
				}
				else {
					if (rt[s,p,t] > -1) {
						log_lik_rt[ix] <- wiener_log(rt[s,p,t], a, ti, 0.5, -v); // For incorrect responses, return lower boundary
					}
					log_lik_resp[ix] <- log(1-p_correct); // Log probability of being wrong
				}
				cf <- cf + feedback[s,p,t]; // increase cumulative feedback by feedback on this trial
                pr_d_c <- exp(binomial_log(cf,t,0.8));
                pr_d_not_c <- exp(binomial_log(cf,t,0.2));
                pr <- (pr_d_c * pri) / ( (pr_d_c * pri) + (pr_d_not_c * (1-pri)) ); // Perceived probability of correct answer being correct, given cumulative feedback
				q[2] <- pr;
				q[1] <- (1 - pr);
			}
		}
	}
}