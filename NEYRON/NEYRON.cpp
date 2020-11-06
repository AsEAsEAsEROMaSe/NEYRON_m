#include <stdarg.h>
#include <iostream>
#include <vector>
#include <omp.h>
#include <fstream>
#include <algorithm>
#include <iterator>
#include <time.h>

using namespace std;
//3 1
/*
from numpy import exp, array, random, dot
training_set_inputs = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
training_set_outputs = array([[0, 1, 1, 0]]).T
random.seed(1)
synaptic_weights = 2 * random.random((3, 1)) - 1
for iteration in xrange(10000):
	output = 1 / (1 + exp(-(dot(training_set_inputs, synaptic_weights))))
	synaptic_weights += dot(training_set_inputs.T, (training_set_outputs - output) * output * (1 - output))
print 1 / (1 + exp(-(dot(array([1, 0, 0]), synaptic_weights))))
*/
/*
 void FindError_End(vector<float>&target)
	{
		for (size_t i = 0; i < size_out; i++)
		{
			loss[i] = 0.0;
		}

		for (size_t i = 0; i < size_out; i++)
		{
			loss[i] = target[i] - out[i];
		}
	}

	void FindError(vector<float>&loss_do)
	{
		for (size_t i = 0; i < size_out; i++)
		{
			loss[i] = 0.0;
		}

		for (size_t i = 0; i < size_out; i++)
		{
			for (size_t j = 0; j < size_in; j++)
			{
				loss[j] += loss_do[i] * w[j][i];
			}
		}
	}

	void BackPropagation()
	{
		for (size_t i = 0; i < size_out; i++)
		{
			for (size_t j = 0; j < size_in; j++)
			{
				w[j][i] += k * loss[j] / w[j][i];// *sigm_pro(out[i])* in[j];
			}
		}
	}

	void cout_mas()
	{
		for ( int i = 0; i < size_out; i++)
		{
			cout << loss[i] << " ";
		}
	}
*/

class Layer
{
public:

	int size_in = 0;
	int size_out = 0;

	const float e = 2.71828182846;

	vector<float>in;
	vector<float>out;
	vector<float>out_non;
	vector<vector<float>>w;
	vector<float>loss;


	inline double rand_d(size_t dip, double size) {
		size_t numb = 0;
		numb = dip;
		double a = 0;
		if (numb != 0) {
			a = (size / numb);
			int n = (rand() % numb) + 1;
			return a * n;
		}
		else {
			return -1;
		}
	}

	inline double rad(double start, double end, size_t znakov_p_k, double v_dipz)
	{
		double ost = rand_d(znakov_p_k, v_dipz);
		size_t a = start; int b = end; int c;
		c = a + rand() % (b - a);
		return (double)c + ost;
	}

	Layer(){}

	Layer(const Layer& L) {
		this->in = L.in;
		this->loss = L.loss;
		this->out = L.out;
		this->out_non = L.out_non;
		this->out_p = L.out_p;
		this->size_in = L.size_in;
		this->size_out = L.size_out;

	/*	for (size_t i = 0; i < size_out; i++)
		{
			w.push_back(vector<float>());
		}

		for (size_t i = 0; i < size_out; i++)
		{
			for (size_t j = 0; j < size_in; j++)
			{
				w[i].push_back(L.w[i][j]);
			}
		}*/
		cout << "Конструктор копирования Layer " << endl;
	}

	Layer& operator = (const Layer& L) {

		Linit(L.size_in, L.size_out);

		//this->in = L.in;
		//this->loss = L.loss;
		//this->out = L.out;
		//this->out_non = L.out_non;
		//this->out_p = L.out_p;

		for (size_t i = 0; i < size_out; i++)
		{
			for (size_t j = 0; j < size_in; j++)
			{
				this->w[i][j] = L.w[i][j];
			}
		}

		cout << "Перегрузка оператора класа Layer " << endl;
		return *this;
	}

	Layer(int input, int output)
	{

		this->size_in = input;
		this->size_out = output;

		for (size_t i = 0; i < size_out; i++)
		{
			loss.push_back(0.0);
			w.push_back(vector<float>());
			out.push_back(0.0);
			out_non.push_back(0.0);
		}

		for (size_t i = 0; i < size_in; i++)
		{
			in.push_back(0.0);
		}

		for (size_t i = 0; i < size_out; i++)
		{
			for (size_t j = 0; j < size_in; j++)
			{
				w[i].push_back(rad(0, 1, 1000, rad(0, 1, 1000, 1)));
			}
		}

	}

	void Linit(int input, int output)
	{

		this->size_in = input;
		this->size_out = output;

		for (size_t i = 0; i < size_out; i++)
		{
			loss.push_back(0.0);
			w.push_back(vector<float>());
			out.push_back(0.0);
			out_non.push_back(0.0);
		}

		for (size_t i = 0; i < size_in; i++)
		{
			in.push_back(0.0);
		}

		for (size_t i = 0; i < size_out; i++)
		{
			for (size_t j = 0; j < size_in; j++)
			{
				w[i].push_back(rad(0, 1, 1000, rad(0, 1, 1000, 1)));
			}
		}

	}

	inline float Sigmoid(float x) {
		return 1 / (1 + pow(e, -x));
	}

	inline double sigm_pro(double x) {
		if ((fabs(x - 1) < 1e-9) || (fabs(x) < 1e-9)) return 0.0;
		double res = x * (1.0 - x);
		return res;
	}
	inline double factanh(double x) {
		return (pow(e, x) - pow(e, (-x))) / (pow(e, x) + pow(e, (-x)));
	}

	inline double funkprotanh(double x) {
		return 1 - pow(x, 2);
	}

	inline double funkAct(double x) {
		return Sigmoid(x);
	}

	inline double funkPro(double x) {
		return sigm_pro(x);
	}

	float out_p = 0.0;

	void FeedForward()
	{
		for (size_t i = 0; i < size_out; i++)
		{
			out_p = 0.0;

			for (size_t j = 0; j < size_in; j++)
			{
				out_p += in[j] * w[i][j];
			}

			//out_non[i] = out_p;
			out[i] = funkAct(out_p);/*sin(out_p);*/ //Sigmoid(out_p);
		}
	}


	void FeedForward(vector<float>&input_v)
	{
		for (size_t i = 0; i < size_out; i++)
		{
			out_p = 0.0;

			for (size_t j = 0; j < size_in; j++)
			{
				out_p += input_v[j] * w[i][j];
				in[j] = input_v[j];
			}

			//out_non[i] = out_p;
			out[i] = funkAct(out_p);///*sin(out_p);*/ Sigmoid(out_p);
		}
	}

	void FeedForward(float inputr) {
		out[0] = inputr * w[0][0];
		out[0] = funkAct(out[0]);// Sigmoid(out[0]); //sin(out[0]);
	}

	void FeedForward(vector<vector<float>>& input_v, int nm)
	{
		for (size_t i = 0; i < size_out; i++)
		{
			out_p = 0.0;

			for (size_t j = 0; j < size_in; j++)
			{
				out_p += input_v[nm][j] * w[i][j];
				in[j] = input_v[nm][j];
			}

			//out_non[i] = out_p;
			out[i] = funkAct(out_p);///*sin(out_p);*/ Sigmoid(out_p);
		}
	}

	void FindError(vector<float>& loss_do)
	{
		for (size_t i = 0; i < size_out; i++)
		{
			loss[i] = 0;
		}

		for (size_t i = 0; i < size_out; i++)
		{
			loss[i] = loss_do[i];
		}
	}

	void FindError_n(vector<float> dov)
	{
		for (size_t i = 0; i < dov.size(); i++)
		{
			dov[i] = 0;
		}

		for (size_t i = 0; i < size_in; i++)
		{
			for (size_t j = 0; j < size_out; j++)
			{
				dov[i] += loss[j] * w[j][i];
			}
		}
	}

	void FindError_End(vector<float>& target)
	{
		for (size_t i = 0; i < size_out; i++)
		{
			loss[i] = target[i] - out[i];
		}
	}

	void FindError_End(vector<vector<float>>& target, int nm)
	{
		int nmm = nm;
		for (size_t i = 0; i < size_out; i++)
		{
			loss[i] = target[nmm][i] - out[i];
		}
	}

	void BackPropagation(float k)
	{
		for (size_t i = 0; i < size_out; i++)
		{
			for (size_t j = 0; j < size_in; j++)
			{
				w[i][j] += k * loss[i] * /*cos(out[i])*//*sigm_pro(out[i])*/funkPro(out[i]) * in[j];
			}
		}
	}

	~Layer() {

	}


private:

};

class NR 
{
public: 

	vector<int> topology;

	vector<vector<float>>in;
	vector<vector<float>>t;

	vector<float>out;

	int primerov = 0;
	int size_layer = 0;
	float k = 0.1;

	std::vector<Layer>N;

	NR(){}

	NR(std::vector<int>& topology) 
	{
		this->topology.clear();
		this->topology = topology;

		N.resize(topology.size() - 1);

		for (size_t i = 0; i < topology.size() - 1; i++)
		{
			N[i].Linit(topology[i], topology[i + 1]);
		}

	}
	NR(std::string name, std::vector<int>& topology) 
	{
		this->topology.clear();
		this->topology = topology;
		N.resize(topology.size() - 1);

		for (size_t i = 0; i < topology.size() - 1; i++)
		{
			N[i].Linit(topology[i], topology[i + 1]);
		}

		LoadObject(name);
	}

	void FeedForward()
	{
		primerov = in.size();
		size_layer = N.size();
		this->out.clear();

		for (size_t j = 0; j < primerov; j++)
		{

			for (size_t z = 0; z < size_layer; z++)
			{
				if (z == 0)
				{
					N[z].FeedForward(in, j);
				}
				else
				{
					N[z].FeedForward(N[z - 1].out);
				}
			}

			for (size_t i = 0; i < N[size_layer - 1].out.size(); i++)
			{
				this->out.push_back(N[size_layer - 1].out[i]);
			}
			
		}
		
	}

	void FeedForward(int prm)
	{
		primerov = in.size();
		size_layer = N.size();
		this->out.clear();

			for (size_t z = 0; z < size_layer; z++)
			{
				if (z == 0)
				{
					N[z].FeedForward(in, prm);
				}
				else
				{
					N[z].FeedForward(N[z - 1].out);
				}
			}

			for (size_t i = 0; i < N[size_layer - 1].out.size(); i++)
			{
				this->out.push_back(N[size_layer - 1].out[i]);
			}

	}

	double loss() {
		int layer_index = N.size() - 1;
		int size_l = N[layer_index].loss.size() - 1;
		double mloss = 0.000000;
		
		if (size_l == 0) {
			mloss = mloss + N[layer_index].loss[0];
		}
		else
		{
			for (size_t i = 0; i < size_l; i++)
			{
				mloss = mloss + N[layer_index].loss[i];
			}
		}

		//cout << endl << "Loss mloss " << mloss << endl;

		if (size_l == 0) {
			size_l = 1;
		}

		if (mloss != NULL) {
			mloss = mloss / size_l;
			if (mloss < 0) {
				mloss = mloss * -1;
			}

			return mloss;
		}
		else
		{
			return 1;
		}
	}

	void Train(int numb_train, std::vector<std::vector<float>>&in, std::vector<std::vector<float>>&t)
	{
		primerov = in.size();
		size_layer = N.size();
		this->in = in;
		this->t = t;

		for (size_t i = 0; i < numb_train; i++)
		{
			for (size_t j = 0; j < primerov; j++)
			{
				for (size_t z = 0; z < size_layer; z++)
				{
					if (z == 0)
					{
						N[z].FeedForward(this->in, j);
					}
					else
					{
						N[z].FeedForward(N[z - 1].out);
					}
				}

				for (size_t z = size_layer - 1; z > 0; z--)
				{
					if (z == size_layer - 1)
					{
						N[z].FindError_End(this->t, j);
						N[z].FindError_n(N[z - 1].loss);
					}
					else
					{
							N[z].FindError_n(N[z - 1].loss);
					}
				}

				for (size_t z = 0; z < size_layer; z++)
				{
					N[z].BackPropagation(k);
				}
			}
		}
	}

	void SaveObject(std::string name) {
		fstream fout;
		fout.open(name , ofstream::app);
		if (!fout.is_open()) {
			std::cout << "EROOR OPEN FILE" << endl;
		}
		else
		{
			float lll;
			std::cout << "FILE OPEN SAVE" << endl;
			/*int top = 0;
			for (size_t i = 0; i < topology.size(); i++)
			{
				top = topology[i];
				fout.write((char*)&top, sizeof(int));
			}*/
			for (size_t i = 0; i < topology.size() - 1; i++) 
			{
				for (size_t ii = 0; ii < N[i].w.size(); ii++)
				{
					for (size_t j = 0; j < N[i].w[ii].size(); j++)
					{
						lll = N[i].w[ii][j];
						fout.write((char*)&lll, sizeof(float));
					}
				}
				
			}
		}
		fout.close();
	}

	void LoadObject(std::string name) {
		ifstream fin;
		fin.open(name);
		if (!fin.is_open()) {
			std::cout << "EROOR OPEN FILE" << endl;
		}
		else
		{
			float lll;
			std::cout << "FILE OPEN LOAD" << endl;
			/*int top = 0;
			while (!fin.eof())
			{
				fin.read((char*)&top, sizeof(int));
				topology.push_back(top);
			}*/
			for (size_t i = 0; i < topology.size() - 1; i++)
			{
				for (size_t ii = 0; ii < N[i].w.size(); ii++)
				{
					for (size_t j = 0; j < N[i].w[ii].size(); j++)
					{
						fin.read((char*)&lll, sizeof(float));
						N[i].w[ii][j] = lll;
					}
				}

			}
		}
		fin.close();
	}
};

inline double rand_d(int dip, double size) {
	int numb = 0;
	numb = dip;
	double a = 0;
	if (numb != 0) {
		a = (size / numb);
		int n = (rand() % numb) + 1;
		return a * n;
	}
	else {
		return -1;
	}
}


int main() 
{
	setlocale(LC_ALL, "Russian");
	srand(time(0));

	vector<vector<float>>in;
	vector<vector<float>>t;
	vector<float>out;
	vector<int>topology;
	
	int sizeN = 0;
	cout << "IN sizeN ";
	cin >> sizeN;

	for (size_t i = 0; i < sizeN; i++)
	{
		int numb = 0;

		cout << "L " << i << " ";
		cin >> numb;

		topology.push_back(numb);
	}

	NR N(topology);

	int primerov = 2;
	float numb = 0.0;
	int train_size = 1;
	float minloss = 0.05;

//#define test

#ifndef test

	cout << endl;
	cout << "Numb primerov ";
	cin >> primerov;

	cout << endl;
	cout << "Numb train_size ";
	cin >> train_size;

#endif // !test

	//cout << endl;
	//cout << endl;

	//cout << endl;
	//cout << "Min loss ";
	//cin >> minloss;
	//cout << endl;

	//cout << endl;
	//cout << "K ";
	//cin >> N.k;
	//cout << endl;*/

	cout << endl;
	for (size_t i = 0; i < primerov; i++)
	{
		in.push_back(vector<float>());
		t.push_back(vector<float>());
	}

	
	for (size_t i = 0; i < primerov; i++)
	{
		for (size_t j = 0; j < topology[0]; j++)
		{
			cout << "IN ";
			cin >> numb;
			in[i].push_back(numb);
		}

		for (size_t j = 0; j < topology[topology.size() - 1]; j++)
		{
			cout << "OUT ";
			cin >> numb;
			t[i].push_back(numb);
		}
		cout << endl;
	}
	cout << endl;
	//tut comentirovati shob test

#ifndef test

	double end = 0.000000000000;
	double start = 0.00000000000;

	if (minloss == NULL) {
		minloss = 0.0155;
	}

	float ls = 0.1;
	float lsd = 0;
	int index_nnn = 0;

	N.k = 0.00001;

	start = omp_get_wtime();

	do
	{
		N.Train(train_size, in, t);
		ls = (float)N.loss();
		cout << endl << "Loss " << ls << " " << endl;
		/*if (ls > lsd) {
			N.k = ls / 2;
			lsd = ls;
		}
		else {
			N.k = ls * 2;
			lsd = ls;
		}*/
		//N.k = 0.00001;
		index_nnn++;

	} while (ls > minloss);
	
	end = omp_get_wtime();

	//cout << "Loss " << N.GetLoss(in, t) << " " << endl;

	cout << endl;
	cout << "Time " << end - start << " sec";
	cout << endl;

	cout << "Epoch " << index_nnn * train_size << endl;

	m1:

	for (size_t j = 0; j < primerov; j++)
	{
		N.FeedForward(j);
		out.clear();
		out = N.out;
		for (size_t i = 0; i < out.size(); i++)
		{
			cout << "PRIMER " << j;
			cout << " OUT " << i << " " << out[i] << " ";
			//cout << "TAR " << i << " " << t[i][0] << " ";
			cout << endl;
		}
		cout << endl;
	}

	//testovi primeri
	in[0].clear();
	for (size_t j = 0; j < topology[0]; j++)
	{
		cout << "IN ";
		cin >> numb;
		if (numb == 0) {
			goto end;
		}
		in[0].push_back(numb);
	}
	primerov = 1;

	goto m1;
	end:
#endif // !test

	//N.SaveObject("obj.txt");

	//out.clear();

	//NR N_2("obj.txt", topology);

	//N_2.in = in;

	//cout << "////////////////////////////////////////" << endl;
	//
	//{
	//	N_2.FeedForward();
	//	out = N_2.out;
	//	//if (out.size() == t.size()) 
	//	{
	//		for (size_t i = 0; i < out.size(); i++)
	//		{
	//			cout << "OUT " << i << " " << out[i] << " ";
	//			cout << endl;
	//		}
	//		cout << endl;
	//	}
	//}

	return 0;
}
//
//int main()
//{
//	/*
//	x_n = r * x_n+1(1 - x_n)
//	float x = 0.1;
//	float r = 2.6;
//
//	for (size_t i = 0; i < 10000000; i++)
//	{
//		x = r * x * (1 - x);
//			cout << x << endl;
//			r = r + 0.0001;
//		cout << One.rand_d(10, 0.8) << endl;
//	}*/
//	
//	Layer One(1, 6);
//	Layer Two(6, 1);
//
//	int primerov = 5;
//
//	vector<vector<float>>in;
//	vector<vector<float>>t;
//
//	vector<float>inv;
//	vector<float>tv;
//
//	vector<float>out;
//
//	float k = 0.1;
//
//	for (size_t i = 0; i < primerov; i++)
//	{
//		in.push_back(vector<float>());
//		t.push_back(vector<float>());
//	}
//
//	float numb = 0.0;
//
//	for (size_t i = 0; i < primerov; i++)
//	{
//		for (size_t j = 0; j < One.size_in; j++)
//		{
//			cout << "IN ";
//			cin >> numb;
//			in[i].push_back(numb);
//		}
//		
//		for (size_t j = 0; j < Two.size_out; j++)
//		{
//			cout << "OUT ";
//			cin >> numb;
//			t[i].push_back(numb);
//		}
//		cout << endl;
//	}
//
//	k = 1;
//	int op = 5000;
//
//	for (size_t i = 0; i < 1000000; i++)
//	{
//		for (size_t j = 0; j < primerov; j++)
//		{
//			One.FeedForward(in, j);
//			Two.FeedForward(One.out);
//			
//			Two.FindError_End(t , j);
//			Two.FindError_n(One.loss);
//
//			One.BackPropagation(k);
//			Two.BackPropagation(k);
//		}
//		
//		if (i == op) {
//			op = op + 5000;
//			for (size_t j = 0; j < primerov; j++)
//			{
//				One.FeedForward(in, j);
//				Two.FeedForward(One.out);
//				out = Two.out;
//				for (size_t i = 0; i < out.size(); i++)
//				{
//					cout << out[i] << " ";
//				}
//				cout << endl;
//			}
//			cout << endl;
//			cout << endl;
//		}
//
//	}
//
//	for (size_t j = 0; j < primerov; j++)
//	{
//		One.FeedForward(in,j);
//		Two.FeedForward(One.out);
//		out = Two.out;
//			for (size_t i = 0; i < out.size(); i++)
//			{
//				cout << out[i] << " ";
//			}
//			cout << endl;
//	}
//	
//
//	return 0;
//}