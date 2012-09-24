%%  plot_number - Plots a 28X28 image of a number
%
%   number: a 1X784 vector representing the number image

function fig = plot_number(number)
    fig = figure();
    
    axis square;
    xlim([1 28]);
    ylim([1 28]);
    hold on;

    for i=0:27
      for j=1:28
        rgb = number(28*i + j);
        if (rgb != 0)
          color = (1 - rgb/256) * [1,1,1];
          plot(j,28-i, "markeredgecolor", color, "markerfacecolor", color, "marker", 'o');
        end
      end
    end

    hold off;
end