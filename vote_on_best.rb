#helper function to find the most common val in array
def most_common_value(a)
  a.group_by do |e|
    e
  end.values.max_by(&:size).first
end

def weight(filename)
  if filename == 'neural_network/neural_network_1_hl_with_200_nodes.csv'
    3
  elsif filename == "knn_classifer/knn_classifier_k20.csv"
    5
  else
    1
  end
end

def main
  #initalize results hash
  path_to_file = File.join('**', '*.csv')
  res = {}
  (0..27999).each {|i| res[i] = [] }

  #copy each file to results hash
  Dir.glob(path_to_file).each do |f| 
    #results from more accurate classifiers are weighted higher
    weight = weight(f)
    unless ['test.csv', 'train.csv'].include? f
      File.open(f).each_line.with_index do |line, index|
        weight.times { res[index] << line.chomp }
      end
    end
  end

  #vote on best, and save to file
  File.open('most_common.csv', 'wb') do |f|
    res.each do |k, array|
      f.puts(most_common_value(array))  
    end
  end
end

main