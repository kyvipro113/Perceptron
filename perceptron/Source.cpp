#include <iostream>
#include <vector>

using namespace std;

class perceptron
{
public:
	perceptron(float eta, int epochs);
	float netInput(vector<float> X);
	int predict(vector<float> X);
	void fit(vector <vector<float> > X, vector<float> y);
	void printError();
	//void exportWeights();
	//void importWeights();
	void printWeights();

private:
	float m_eta;
	int m_epochs;
	vector <float> m_w;
	vector <float> m_errors;
};

perceptron::perceptron(float eta, int epochs)
{
	this->m_eta = eta;
	this->m_epochs = epochs;
}

int perceptron::predict(vector<float> X)
{
	return netInput(X) > 0 ? 1 : 0;
}

float perceptron::netInput(vector<float> X)
{
	float probabilities = this->m_w[0];
	for (int i = 0; i < X.size(); i++)
	{
		probabilities += X[i] * this->m_w[i + 1];
	}
	return probabilities;
}

void perceptron::fit(vector<vector<float> > X, vector<float> y)
{
	for (int i = 0; i < X[0].size() + 1; i++)
	{
		this->m_w.push_back(0);
	}
	for (int i = 0; i < this->m_epochs; i++)
	{
		int errors = 0;
		for (int j = 0; j < X.size(); j++)
		{
			float update = this->m_eta * (y[j] - predict(X[j]));
			for (int w = 1; w < this->m_w.size(); w++)
			{
				this->m_w[w] += update * X[j][w - 1];
			}
			this->m_w[0] = update;
			errors += update != 0 ? 1 : 0;
		}
		this->m_errors.push_back(errors);
	}
}

void perceptron::printError()
{
	/*for (vector <float>::const_iterator item = this->m_errors.begin(); item != this->m_errors.end(); item++)
		cout << *item << " ";*/
	for (int i = 0; i < this->m_errors.size(); i++)
		cout << "Epoch Error[" << i + 1 << "]: " << this->m_errors[i] << endl;
	cout << endl;
}

void perceptron::printWeights()
{
	cout << "W[0]: " << this->m_w[0] << endl;
	for (int i = 1; i < this->m_w.size(); i++)
		cout << "W[" << i << "]: " << this->m_w[i] << endl;
}

int main()
{
	vector < vector <float> > X_train = { {1, 2, 3, 4, 5},
										  {6, 7, 8, 9, 0},
										  {4, 5, 6, 7, 8},
										  {0, 1, 6, 9, 8},
										  {9, 1, 0, 3, 4},
										  {5, 6, 7, 8, 6} };
	vector <float> y_train = {1, 1, 0, 1, 0, 1};

	vector <float> X_test = { 1, 2, 2, 4, 5 };

	float eta = 0.01;
	int epochs = 7;

	perceptron neural = perceptron(eta, epochs);
	neural.fit(X_train, y_train);
	neural.printWeights();
	cout << endl;
	neural.printError();
	cout << endl;
	int y_pred = neural.predict(X_test);
	cout << "y predict: " << y_pred << endl;
	system("pause");
	return 0;
}