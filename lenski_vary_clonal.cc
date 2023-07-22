#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/types.h>
#include <sys/stat.h>
#include <time.h>
#include <omp.h>
#include <iomanip>
#include <sstream>

#include "lenski_sim.hh"

void output_sim_info(int L, 
                     int N_0, 
                     int N_f, 
                     int ndays, 
                     double dt, 
                     double mu, 
                     string base_folder, 
                     double rho, 
                     double beta, 
                     double delta, 
                     int init_rank, 
                     int rank_interval, 
                     bool hoc) {

    string out_str = base_folder + "/sim_data.txt";
    FILE *outf = fopen(out_str.c_str(), "w");
    if (outf == NULL) {
        fprintf(stderr, "Error opening file %s\n.", out_str.c_str());
    }

    fprintf(outf, "L: %d\n", L);
    fprintf(outf, "N_0: %d\n", N_0);
    fprintf(outf, "N_f: %d\n", N_f);
    fprintf(outf, "ndays: %d\n", ndays);
    fprintf(outf, "dt: %g\n", dt);
    fprintf(outf, "mu: %g\n", mu);
    fprintf(outf, "rho: %g\n", rho);
    fprintf(outf, "beta: %g\n", beta);
    fprintf(outf, "delta: %g\n", delta);
    fprintf(outf, "init_rank: %d\n", init_rank);
    fprintf(outf, "rank_interval: %d\n", rank_interval);
    fprintf(outf, "hoc: %d\n", hoc);
    fclose(outf);
}


// Describes the command line argument structure in the case of an incorrect 
// set of parameters.
void syntax_message(int argc) {
    printf("\nGot %d arguments. Expected 16.\n", argc);
	printf("Syntax: ./lenski_vary_epi L N_0 N_f n_outputs n_outputs_rank nexps dt mu base_folder init_rank rho delta HOC\n\n");
    printf("L: Size of genome. \n");
    printf("N_0: Initial number of bacteria. \n");
    printf("N_f: Number of bacteria at end of day. \n");
    printf("n_outputs: Number of output points for bacteria/mutation info. \n");
    printf("n_outputs_rank: Number of output points for rank/mechanism info. \n");
    printf("nexps: Number of experiments (individual simulations) to simulate.. \n");
    printf("dt: Timestep. \n");
    printf("beta: Value of beta.\n");
    printf("base_folder: Name of the folder containing replicate simulation data. \n");
    printf("init_rank: Rank of the initial strain? -1 if you do not care, and want it to be random.\n");
    printf("rho: Density of interaction matrix. \n");
    printf("delta: controls fitness effects of mutations. \n");
    printf("HOC: Whether or not to use the house of cards at each gene model.\n");
	exit(1);
}


int main(int argc, char **argv) {
	// check command-line arguments.
	if (argc != 14) syntax_message(argc);

    // pick off the command line arguments.
    int L               = (int) std::stod(argv[1]);
    int N_0             = (int) std::stod(argv[2]);
    int N_f             = (int) std::stod(argv[3]);
    int n_outputs       = std::stoi(argv[4]);
    int n_outputs_rank  = std::stoi(argv[5]);
    int nexps           = std::stoi(argv[6]);
    double dt           = std::stod(argv[7]);
    double beta         = std::stod(argv[8]);
    string base_folder  = argv[9];
    int init_rank       = (int) std::stod(argv[10]);
    double rho          = std::stod(argv[11]);
    double delta        = std::stod(argv[12]);
    bool hoc            = (bool) std::stoi(argv[13]);

    // epistasis information
    double sigh = sqrt((1-beta))*delta;
    double muJ  = 0;
    double sigJ = sqrt(beta)*delta/sqrt(L*rho)/2;
    bool interact = beta > 0;

    // binning information.
    int nbins = 250;
    double min_select = -.125;
    double max_select = .125;
    vector<double> mus = {1e-8, 1e-7, 1e-6, 1e-5, 1e-4};
    vector<int> ndays = {500000000, 50000000, 5000000, 2000000, 1000000};

    vector<int> output_intervals; vector<int> rank_intervals;
    vector<string> base_folders;

    // draw seeds for the random number generators
    vector<uint32_t> seeds;
    std::random_device rd;
    for (unsigned int ii = 0; ii < nexps*mus.size(); ii++) { seeds.push_back(rd()); }

    // set up the mu-dependent parameters
    string new_base_folder;
    int output_interval, rank_interval;
    for (unsigned int ii = 0; ii < mus.size(); ii++) {
        // output information
        output_interval = ndays[ii]/n_outputs; 
        output_intervals.push_back(output_interval);
        rank_interval = ndays[ii]/n_outputs_rank; 
        rank_intervals.push_back(rank_interval);

        // save the string defining the output for this class of simulations
        std::stringstream mu_val_stream;
        mu_val_stream << std::fixed << std::setprecision(10) << mus[ii];
        new_base_folder = base_folder + "_mu" + mu_val_stream.str();
        base_folders.push_back(new_base_folder);

        // output simulation info for this batch of replicates
        mkdir(new_base_folder.c_str(), 0700);
        output_sim_info(L, N_0, N_f, ndays[ii], dt, mus[ii], new_base_folder, 
                        rho, beta, delta, init_rank, rank_interval, hoc);
    }


    // set up and save the base simulation to copy the disorder from
    // note that when varying the mutation rate, we only need a single
    // base simulation class (unlike when varying the epistasis parameters)
    // so that every simulation takes place in the same landscape.
    lenski_sim base_sim = lenski_sim(L, N_0, N_f, mus[0]/L, rho, sigh, muJ, sigJ, 
                                     new_base_folder, interact, ndays[0]/n_outputs, 
                                     init_rank, ndays[0]/n_outputs_rank, nbins, 
                                     min_select, max_select, seeds[0], hoc, 0, false);

    // compute fixed initializations for each replicate
    vector<vector<double>> alpha0s_and_Jalpha0s = base_sim.compute_alpha0s_and_Jalpha0s(nexps);

// now parallelize over all values of mu and all replicates
#pragma omp parallel for collapse(2)
    for (unsigned int jj = 0; jj < mus.size(); jj++) {
        for (int curr_replicate = 0; curr_replicate < nexps; curr_replicate++) {
            // copy over the quenched disorder and fixed initial sequences.
            // pass -1 for rank argument to initialize from same locations, 
            // pass init_rank to initialize from different locations.
            int kk = curr_replicate + jj*mus.size();
            double p = mus[jj] / L;
            lenski_sim replicate_sim(L, N_0, N_f, p, rho, sigh, muJ, 
                                     sigJ, base_folders[jj], 
                                     interact, output_intervals[jj], 
                                     -1, rank_intervals[jj], nbins, 
                                     min_select, max_select, seeds[kk], hoc, 
                                     curr_replicate, false, 1000);
            
            replicate_sim.copy_disorder(base_sim.get_his(), 
                                        base_sim.get_Jijs(), 
                                        alpha0s_and_Jalpha0s[2*curr_replicate], 
                                        alpha0s_and_Jalpha0s[2*curr_replicate+1]);

            // run and time the sim
            time_t start_t(0), end_t(0);
            time(&start_t); 
            replicate_sim.simulate_experiment(ndays[jj], dt); 
            time(&end_t);
            printf("Finished replicate %d for mu=%0.9f! Total time: %.21f hours.\n", 
                   curr_replicate, p*L, difftime(end_t, start_t)/3600.);
        }
    }

    return (0);
}
