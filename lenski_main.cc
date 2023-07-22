#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/types.h>
#include <sys/stat.h>
#include <time.h>
#include <omp.h>

#include "lenski_sim.hh"


void output_sim_info(
        int L, 
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


// Describes the command line argument structure in the case of an 
// incorrect set of parameters.
void syntax_message(int argc) {
    printf("\nGot %d arguments. Expected 16.\n", argc);
	printf("Syntax: ./lenski_small L N_0 N_f ndays nexps dt p_val \
            output_interval base_folder interact init_rank rank_fac \
            rho beta delta HOC\n\n");
    printf("L: Size of genome. \n");
    printf("N_0: Initial number of bacteria. \n");
    printf("N_f: Number of bacteria at end of day. \n");
    printf("ndays: Number of days to simulate. \n");
    printf("nexps: Number of experiments (individual simulations) \
            to simulate.. \n");
    printf("dt: Timestep. \n");
    printf("p_val: Value of p, input as mutation probability \
            per division (p*L). \n");
    printf("output_interval: Output data every output_interval \
            number of days. \n");
    printf("base_folder: Name of the folder containing replicate \
            simulation data. \n");
    printf("init_rank: Rank of the initial strain? -1 if you do not \
            care, and want it to be random.\n");
    printf("rank_interval: Compute the rank every rank_interval days.\n");
    printf("rho: Density of interaction matrix. \n");
    printf("beta: epistatic contribution. \n");
    printf("delta: controls fitness effects of mutations. \n");
    printf("HOC: Whether or not to use the house of cards at \
            each genome model.\n");
	exit(1);
}

int main(int argc, char **argv) {
	// Check command-line arguments.
	if (argc != 16) syntax_message(argc);

    // Pick off the command line arguments.
    int L               = (int) std::stod(argv[1]);
    int N_0             = (int) std::stod(argv[2]);
    int N_f             = (int) std::stod(argv[3]);
    int ndays           = (int) std::stod(argv[4]);
    int nexps           = std::stoi(argv[5]);
    double dt           = std::stod(argv[6]);
    double p            = std::stod(argv[7])/L;
    int output_interval = std::stod(argv[8]);
    string base_folder  = argv[9];
    int init_rank       = (int) std::stod(argv[10]);
    int rank_interval   = std::stoi(argv[11]);
    double rho          = std::stod(argv[12]);
    double beta         = std::stod(argv[13]);
    double delta        = std::stod(argv[14]);
    bool hoc            = (bool) std::stoi(argv[15]);

    printf("L: %d N_0: %d N_f: %d ndays: %d dt: %f hoc: %d\n", 
            L, N_0, N_f, ndays, dt, hoc);

    double sigh = sqrt((1-beta))*delta;
    double muJ  = 0;
    double sigJ = sqrt(beta)*delta/sqrt(L*rho)/2;
    int nbins = 250;
    double min_select = -.125;
    double max_select = .125;
    bool interact = beta > 0;

    // make the output directory and save simulation info
    mkdir(base_folder.c_str(), 0700);
    output_sim_info(L, N_0, N_f, ndays, dt, p*L, base_folder, rho, 
                    beta, delta, init_rank, rank_interval, hoc);

    // draw seeds for the random number generators
    std::random_device rd; vector<uint32_t> seeds;
    for (int curr_replicate = 0; curr_replicate < nexps; curr_replicate++) { 
        seeds.push_back(rd()); 
    }

    // construct a base simulation environment that all replicates are 
    // defined with respect to.
    // this ensures that the initial sequence and quenched disorder are 
    // identical for each replicate.
    lenski_sim base_sim(L, N_0, N_f, p, rho, sigh, muJ, sigJ, base_folder, 
                        interact, output_interval, init_rank, 
                        rank_interval == -1? ndays : rank_interval, nbins, 
                        min_select, max_select, seeds[0], hoc, 0, false, 10);

#pragma omp parallel for
    for (int curr_replicate = 0; curr_replicate < nexps; curr_replicate++) {
        time_t start_t(0), end_t(0);
        printf("Starting replicate %d.\n", curr_replicate);

        // copy over the quenched disorder
        lenski_sim replicate_sim(L, N_0, N_f, p, rho, sigh, muJ, sigJ, 
                                 base_folder, interact, output_interval, -1, 
                                 rank_interval == -1? ndays : rank_interval, 
                                 nbins, min_select, max_select, 
                                 seeds[curr_replicate], hoc, 
                                 curr_replicate, false, 10);

        replicate_sim.copy_disorder(base_sim.get_his(), 
                                    base_sim.get_Jijs(), 
                                    base_sim.get_alpha0s(), 
                                    base_sim.get_Jalpha0());

        time(&start_t); 
        replicate_sim.simulate_experiment(ndays, dt); 
        time(&end_t);
        printf("Finished replicate %d! Total time: %.21f hours.\n", 
                curr_replicate, difftime(end_t, start_t)/3600.);
    }

    return (0);
}
